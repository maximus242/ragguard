"""
Utility functions used throughout the RAGGuard library.

This module provides common helper functions for string manipulation,
value extraction, and other operations used across multiple modules.
"""

import hmac
import re
from typing import Any

from .constants import (
    DOCUMENT_PREFIX,
    DOCUMENT_PREFIX_LEN,
    USER_PREFIX,
    USER_PREFIX_LEN,
)


def strip_user_prefix(field: str) -> str:
    """
    Remove the 'user.' prefix from a field name if present.

    Args:
        field: Field name that may start with 'user.'

    Returns:
        Field name without the 'user.' prefix

    Examples:
        >>> strip_user_prefix("user.department")
        'department'
        >>> strip_user_prefix("user.roles")
        'roles'
        >>> strip_user_prefix("department")
        'department'
    """
    if field.startswith(USER_PREFIX):
        return field[USER_PREFIX_LEN:]
    return field


def strip_document_prefix(field: str) -> str:
    """
    Remove the 'document.' prefix from a field name if present.

    Args:
        field: Field name that may start with 'document.'

    Returns:
        Field name without the 'document.' prefix

    Examples:
        >>> strip_document_prefix("document.category")
        'category'
        >>> strip_document_prefix("document.metadata.tags")
        'metadata.tags'
        >>> strip_document_prefix("category")
        'category'
    """
    if field.startswith(DOCUMENT_PREFIX):
        return field[DOCUMENT_PREFIX_LEN:]
    return field


def strip_field_prefix(field: str) -> tuple[str, str]:
    """
    Remove either 'user.' or 'document.' prefix and return the prefix type.

    Args:
        field: Field name that may start with 'user.' or 'document.'

    Returns:
        Tuple of (stripped_field, prefix_type) where prefix_type is
        'user', 'document', or 'unknown'

    Examples:
        >>> strip_field_prefix("user.department")
        ('department', 'user')
        >>> strip_field_prefix("document.category")
        ('category', 'document')
        >>> strip_field_prefix("other_field")
        ('other_field', 'unknown')
    """
    if field.startswith(USER_PREFIX):
        return field[USER_PREFIX_LEN:], "user"
    elif field.startswith(DOCUMENT_PREFIX):
        return field[DOCUMENT_PREFIX_LEN:], "document"
    return field, "unknown"


def is_user_field(field: str) -> bool:
    """Check if a field references user context."""
    return field.startswith(USER_PREFIX)


def is_document_field(field: str) -> bool:
    """Check if a field references document metadata."""
    return field.startswith(DOCUMENT_PREFIX)


def get_nested_value(
    obj: dict[str, Any],
    path: str,
    default: Any = None
) -> Any:
    """
    Get a value from a nested dictionary using dot notation.

    Args:
        obj: The dictionary to search
        path: Dot-separated path (e.g., 'metadata.author.name')
        default: Value to return if path not found

    Returns:
        The value at the path, or default if not found

    Examples:
        >>> data = {"user": {"profile": {"name": "Alice"}}}
        >>> get_nested_value(data, "user.profile.name")
        'Alice'
        >>> get_nested_value(data, "user.email", "unknown")
        'unknown'
    """
    if not obj or not path:
        return default

    parts = path.split(".")
    current = obj

    for part in parts:
        if isinstance(current, dict):
            if part in current:
                current = current[part]
            else:
                return default
        else:
            return default

    return current


def set_nested_value(
    obj: dict[str, Any],
    path: str,
    value: Any
) -> None:
    """
    Set a value in a nested dictionary using dot notation.

    Args:
        obj: The dictionary to modify
        path: Dot-separated path (e.g., 'metadata.author.name')
        value: The value to set

    Examples:
        >>> data = {}
        >>> set_nested_value(data, "user.profile.name", "Alice")
        >>> data
        {'user': {'profile': {'name': 'Alice'}}}
    """
    if not path:
        return

    parts = path.split(".")
    current = obj

    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]

    current[parts[-1]] = value


def parse_literal_value(value_str: str) -> Any:
    """
    Parse a string literal value into its Python type.

    Handles strings, numbers, booleans, None, and lists.

    Args:
        value_str: String representation of a value

    Returns:
        Parsed Python value

    Examples:
        >>> parse_literal_value("'hello'")
        'hello'
        >>> parse_literal_value("42")
        42
        >>> parse_literal_value("true")
        True
        >>> parse_literal_value("['a', 'b']")
        ['a', 'b']
    """
    value_str = value_str.strip()

    # Handle quoted strings
    if (value_str.startswith("'") and value_str.endswith("'")) or \
       (value_str.startswith('"') and value_str.endswith('"')):
        return value_str[1:-1]

    # Handle booleans
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False

    # Handle None
    if value_str.lower() in ("none", "null"):
        return None

    # Handle numbers
    try:
        # Check for float indicators: decimal point or scientific notation (e/E)
        if "." in value_str or "e" in value_str.lower():
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass

    # Handle lists
    if value_str.startswith("[") and value_str.endswith("]"):
        return parse_list_literal(value_str)

    # Return as-is (field reference or unknown)
    return value_str


def parse_list_literal(list_str: str) -> list[Any]:
    """
    Parse a list literal string into a Python list.

    Args:
        list_str: String representation of a list like "['a', 'b', 'c']"

    Returns:
        Parsed Python list

    Examples:
        >>> parse_list_literal("['cs.AI', 'cs.LG']")
        ['cs.AI', 'cs.LG']
        >>> parse_list_literal("[1, 2, 3]")
        [1, 2, 3]
    """
    if not list_str.startswith("[") or not list_str.endswith("]"):
        return []

    inner = list_str[1:-1].strip()
    if not inner:
        return []

    # Use regex to properly split while respecting quotes
    pattern = r'''(?:[^,'"]+|'[^']*'|"[^"]*")+'''
    items = re.findall(pattern, inner)

    result = []
    for item in items:
        item = item.strip()
        if item:
            result.append(parse_literal_value(item))

    return result


def sanitize_field_name(field: str) -> str:
    """
    Sanitize a field name for safe use in database queries.

    Removes or escapes potentially dangerous characters.

    Args:
        field: The field name to sanitize

    Returns:
        Sanitized field name

    Raises:
        ValueError: If field name contains invalid characters
    """
    # Allow alphanumeric, underscores, dots (for nested), and hyphens
    if not re.match(r'^[\w.\-]+$', field):
        raise ValueError(f"Invalid characters in field name: {field}")

    # Prevent SQL injection patterns
    dangerous_patterns = [
        "--",
        ";",
        "/*",
        "*/",
        "@@",
        "char(",
        "nchar(",
        "varchar(",
        "nvarchar(",
        "alter ",
        "begin ",
        "cast(",
        "create ",
        "cursor ",
        "declare ",
        "delete ",
        "drop ",
        "end ",
        "exec ",
        "execute ",
        "fetch ",
        "insert ",
        "kill ",
        "open ",
        "select ",
        "sys.",
        "sysobjects",
        "syscolumns",
        "table ",
        "update ",
    ]

    field_lower = field.lower()
    for pattern in dangerous_patterns:
        if pattern in field_lower:
            raise ValueError(f"Potentially dangerous pattern in field name: {field}")

    return field


def truncate_string(s: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length with suffix.

    Args:
        s: String to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to append when truncated

    Returns:
        Truncated string

    Examples:
        >>> truncate_string("Hello World", 8)
        'Hello...'
    """
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


def format_duration_ms(duration_seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable millisecond string.

    Args:
        duration_seconds: Duration in seconds

    Returns:
        Formatted string like "123.45ms"

    Examples:
        >>> format_duration_ms(0.12345)
        '123.45ms'
    """
    return f"{duration_seconds * 1000:.2f}ms"


def is_valid_policy_version(version: str) -> bool:
    """
    Check if a policy version string is valid.

    Currently supports version "1".

    Args:
        version: Version string to validate

    Returns:
        True if valid, False otherwise
    """
    return version in ("1", "1.0")


def deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base dictionary
        override: Dictionary with values to override

    Returns:
        New merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def secure_compare(a: Any, b: Any) -> bool:
    """
    Perform constant-time comparison of two values.

    This prevents timing attacks by ensuring the comparison takes the same
    amount of time regardless of where the values differ. This is important
    for security-sensitive operations like permission checks.

    For strings, uses hmac.compare_digest which is designed for constant-time
    comparison. For other types, falls back to regular comparison (which is
    still generally constant-time for fixed-size types like numbers and bools).

    Args:
        a: First value to compare
        b: Second value to compare

    Returns:
        True if values are equal, False otherwise

    Security Note:
        Standard Python string comparison (`==`) returns as soon as it finds
        a difference. An attacker could measure response times to deduce
        information about expected values. This function mitigates that risk.

    Example:
        >>> secure_compare("admin", "admin")
        True
        >>> secure_compare("admin", "user")
        False
        >>> secure_compare(123, 123)
        True
    """
    # Handle None values
    if a is None or b is None:
        # Can't use constant-time comparison with None
        # Both being None is typically a security concern anyway
        return False

    # For strings, use constant-time comparison
    if isinstance(a, str) and isinstance(b, str):
        # hmac.compare_digest requires bytes or ASCII strings
        try:
            return hmac.compare_digest(a.encode('utf-8'), b.encode('utf-8'))
        except (UnicodeEncodeError, TypeError):
            # Fallback for edge cases (shouldn't happen with valid strings)
            return a == b

    # For bytes, use constant-time comparison directly
    if isinstance(a, bytes) and isinstance(b, bytes):
        return hmac.compare_digest(a, b)

    # For lists, compare each element (for role/group membership checks)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        # Compare all elements, but accumulate result to avoid early exit
        result = True
        for x, y in zip(a, b):
            if not secure_compare(x, y):
                result = False
        return result

    # For other types (int, float, bool, etc.), regular comparison is fine
    # These are fixed-size types where comparison time doesn't leak information
    return a == b


def secure_contains(value: Any, collection: list) -> bool:
    """
    Check if a value is in a collection using constant-time comparison.

    Prevents timing attacks when checking membership in sensitive collections
    like role lists or access control groups.

    Args:
        value: Value to search for
        collection: List to search in

    Returns:
        True if value is in collection, False otherwise

    Example:
        >>> secure_contains("admin", ["user", "admin", "guest"])
        True
        >>> secure_contains("root", ["user", "admin", "guest"])
        False
    """
    if not isinstance(collection, list):
        return False

    # Check all items and accumulate result to prevent timing leak
    found = False
    for item in collection:
        if secure_compare(value, item):
            found = True
    return found
