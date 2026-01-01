"""
Shared utilities for filter builders.

Contains common functions used across all database backend filter builders.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

from ..constants import (
    DENY_ALL_FIELD,
    DENY_ALL_VALUE,
    DOCUMENT_PREFIX,
    SKIP_RULE,
    USER_PREFIX,
)
from ..policy.models import AllowConditions
from ..utils import (
    is_document_field,
    is_user_field,
    strip_document_prefix,
    strip_user_prefix,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Re-export constants for backend modules
__all__ = [
    # Shared functions
    "user_satisfies_allow",
    "get_nested_value",
    "parse_literal_value",
    "parse_list_literal",
    "validate_sql_identifier",
    "validate_field_name",
    "validate_field_path",
    # Re-exported constants
    "DENY_ALL_FIELD",
    "DENY_ALL_VALUE",
    "USER_PREFIX",
    "DOCUMENT_PREFIX",
    "SKIP_RULE",
    # Re-exported utils
    "strip_user_prefix",
    "strip_document_prefix",
    "is_user_field",
    "is_document_field",
    "logger",
]


def user_satisfies_allow(allow: AllowConditions, user: dict[str, Any]) -> bool:
    """
    Check if a user satisfies the basic allow conditions (roles, everyone).

    Args:
        allow: The allow conditions from a rule
        user: User context dictionary

    Returns:
        True if user satisfies role or everyone conditions
    """
    if allow.everyone is True:
        return True

    if allow.roles:
        user_roles = user.get("roles", [])
        if user_roles is None:
            user_roles = []
        elif isinstance(user_roles, str):
            user_roles = [user_roles]
        elif isinstance(user_roles, list):
            if not all(isinstance(role, str) for role in user_roles):
                raise ValueError(
                    f"Invalid user.roles: list elements must be strings, got {[type(r).__name__ for r in user_roles]}"
                )
        else:
            raise ValueError(
                f"Invalid user.roles type: expected list or string, got {type(user_roles).__name__}"
            )

        # Limit total number of roles to prevent DoS attacks
        MAX_ROLES = 100
        if len(user_roles) > MAX_ROLES:
            raise ValueError(
                f"Invalid user.roles: too many roles ({len(user_roles)}, max {MAX_ROLES}). "
                f"This limit prevents denial-of-service attacks."
            )

        for role in user_roles:
            if not role:
                raise ValueError("Invalid user.roles: role cannot be empty string")
            if len(role) > 100:
                raise ValueError(f"Invalid user.roles: role too long ({len(role)} chars, max 100): '{role[:50]}...'")
            if not all(c.isalnum() or c in ('-', '_', '.', ':', '@', '/') for c in role):
                invalid_chars = [c for c in role if not (c.isalnum() or c in ('-', '_', '.', ':', '@', '/'))]
                raise ValueError(
                    f"Invalid user.roles: role contains invalid characters. "
                    f"Only alphanumeric and -_.:/@ are allowed. Invalid chars: {invalid_chars} in role: '{role}'"
                )

        if any(role in allow.roles for role in user_roles):
            return True

    if allow.conditions and not allow.roles and not allow.everyone:
        return True

    return False


def get_nested_value(obj: dict[str, Any], key: str) -> Any:
    """
    Get a value from a nested dictionary using dot notation.

    Args:
        obj: Dictionary to extract value from
        key: Dot-separated path (e.g., "user.department")

    Returns:
        Value at the path, or None if not found
    """
    keys = key.split(".")
    value = obj

    for k in keys:
        if isinstance(value, dict):
            value = value.get(k)
            if value is None:
                return None
        else:
            return None

    return value


def parse_literal_value(expr: str) -> Any:
    """
    Parse a literal value from an expression string.

    Handles:
    - Quoted strings: 'public' or "public" -> "public"
    - Numbers: 42 -> 42, 3.14 -> 3.14
    - Booleans: true/false -> True/False
    - null/None -> None

    Args:
        expr: Expression string to parse

    Returns:
        Parsed value (string, int, float, bool, or None)
    """
    expr = expr.strip()

    if expr.lower() in ("none", "null"):
        return None

    if (expr.startswith('"') and expr.endswith('"')) or \
       (expr.startswith("'") and expr.endswith("'")):
        return expr[1:-1]

    if expr.lower() == "true":
        return True
    if expr.lower() == "false":
        return False

    try:
        # Check for float indicators: decimal point or scientific notation (e/E)
        if "." in expr or "e" in expr.lower():
            num_value = float(expr)

            import math
            if math.isnan(num_value):
                raise ValueError(
                    "Float literal cannot be NaN (not a number). "
                    "Use explicit null checks instead."
                )
            if math.isinf(num_value):
                raise ValueError(
                    "Float literal cannot be infinity. "
                    "Use explicit comparisons instead."
                )

            MAX_SAFE_FLOAT = 1.7976931348623157e+308
            if abs(num_value) > MAX_SAFE_FLOAT:
                raise ValueError(f"Float literal too large: {num_value}")
            return num_value
        else:
            num_value = int(expr)
            MAX_SAFE_INT = 9007199254740991
            if abs(num_value) > MAX_SAFE_INT:
                raise ValueError(f"Integer literal too large: {num_value}")
            return num_value
    except ValueError:
        pass

    return expr


def parse_list_literal(expr: str) -> Optional[list[Any]]:
    """
    Parse a list literal from an expression string.

    Handles:
    - ['a', 'b', 'c'] -> ['a', 'b', 'c']
    - [1, 2, 3] -> [1, 2, 3]
    - ['cs.AI', 'cs.LG'] -> ['cs.AI', 'cs.LG']

    Args:
        expr: Expression string to parse

    Returns:
        List of parsed values, or None if not a valid list literal

    Raises:
        ValueError: If the list literal is malformed
    """
    expr = expr.strip()

    # Check if it's a list literal
    if not expr.startswith('['):
        return None

    # Check for missing closing bracket
    if not expr.endswith(']'):
        bracket_pos = expr.index('[')
        raise ValueError(
            f"Malformed list literal: missing closing bracket ']'\n"
            f"  {expr}\n"
            f"  {' ' * bracket_pos}^\n"
            f"  Expected ']' at end of list"
        )

    # Extract content between brackets
    content = expr[1:-1].strip()

    # Empty list
    if not content:
        return []

    # Check for nested lists (simple heuristic)
    if '[' in content or ']' in content:
        raise ValueError(
            f"Nested lists are not supported\n"
            f"  Found in: {expr}\n"
            f"  List elements must be simple values (strings, numbers, booleans)"
        )

    # Check for unclosed quotes using proper quote-aware parsing
    in_single_quote = False
    in_double_quote = False
    i = 0
    while i < len(content):
        char = content[i]

        # Handle escape sequences
        if char == '\\' and i + 1 < len(content):
            i += 2
            continue

        # Track quote state
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote

        i += 1

    # Check for unclosed quotes
    if in_single_quote:
        raise ValueError(
            f"Malformed list literal: unclosed single quote\n"
            f"  {expr}\n"
            f"  Check that all strings are properly quoted"
        )
    if in_double_quote:
        raise ValueError(
            f"Malformed list literal: unclosed double quote\n"
            f"  {expr}\n"
            f"  Check that all strings are properly quoted"
        )

    # Split by comma and parse each item
    items = []
    try:
        for item in content.split(','):
            item = item.strip()
            if item:
                items.append(parse_literal_value(item))
    except Exception as e:
        raise ValueError(
            f"Error parsing list literal: {e!s}\n"
            f"  In expression: {expr}\n"
            f"  Each element should be a quoted string, number, or boolean"
        )

    return items


def validate_sql_identifier(value: str, param_name: str) -> None:
    """
    Validate SQL identifier to prevent SQL injection.

    Args:
        value: The identifier to validate
        param_name: Name of the parameter (for error messages)

    Raises:
        ValueError: If the identifier contains invalid characters
    """
    if not value or not isinstance(value, str):
        raise ValueError(f"{param_name} must be a non-empty string")

    if len(value) > 63:  # PostgreSQL identifier length limit
        raise ValueError(f"{param_name} too long: {len(value)} chars (max 63)")

    # Allow alphanumeric, underscore, and dollar sign (PostgreSQL standard)
    # Must start with letter or underscore
    if not value[0].isalpha() and value[0] != '_':
        raise ValueError(
            f"Invalid {param_name}: must start with letter or underscore, got '{value[0]}'"
        )

    if not all(c.isalnum() or c in ('_', '$') for c in value):
        raise ValueError(
            f"Invalid {param_name}: contains invalid characters. "
            f"Only alphanumeric, underscore, and $ are allowed. Got: '{value}'"
        )


def validate_field_name(field: str, backend: str = "generic") -> str:
    """
    Validate field name for safe use in filter queries.

    Prevents injection attacks by ensuring field names contain only safe characters.
    This function should be called before interpolating any field name into a query.

    Args:
        field: The field name to validate (may contain dots for nested fields)
        backend: Backend name for error messages (default: "generic")

    Returns:
        The validated field name (unchanged if valid)

    Raises:
        ValueError: If the field name is invalid or contains unsafe characters
    """
    if not field or not isinstance(field, str):
        raise ValueError(f"Invalid field name for {backend}: field must be a non-empty string")

    # Check length limit
    if len(field) > 256:
        raise ValueError(
            f"Invalid field name for {backend}: field name too long ({len(field)} chars, max 256)"
        )

    # Check for empty after strip
    field_stripped = field.strip()
    if not field_stripped:
        raise ValueError(f"Invalid field name for {backend}: field cannot be whitespace only")

    # Allow only safe characters: alphanumeric, underscore, dot (for nesting)
    # Must start with a letter or underscore
    import re
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_.]*$', field):
        raise ValueError(
            f"Invalid field name for {backend}: field '{field}' contains unsafe characters. "
            f"Only letters, numbers, underscores, and dots are allowed, "
            f"and must start with a letter or underscore."
        )

    # Check that dots are not at start/end and no double dots
    if field.startswith('.') or field.endswith('.') or '..' in field:
        raise ValueError(
            f"Invalid field name for {backend}: field '{field}' has invalid dot placement"
        )

    return field


def validate_field_path(field_path: list[str], backend: str = "generic") -> str:
    """
    Validate and join a field path for safe use in filter queries.

    Args:
        field_path: List or tuple of field path components (e.g., ['user', 'department'])
        backend: Backend name for error messages

    Returns:
        Dot-joined field name after validation

    Raises:
        ValueError: If any component is invalid
    """
    if not field_path or not isinstance(field_path, (list, tuple)):
        raise ValueError(f"Invalid field path for {backend}: must be a non-empty list or tuple")

    for i, component in enumerate(field_path):
        if not component or not isinstance(component, str):
            raise ValueError(
                f"Invalid field path for {backend}: component {i} must be a non-empty string"
            )
        # Each component must be a valid identifier (no dots allowed in individual components)
        import re
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', component):
            raise ValueError(
                f"Invalid field path for {backend}: component '{component}' contains unsafe characters. "
                f"Only letters, numbers, and underscores are allowed."
            )
        if len(component) > 64:
            raise ValueError(
                f"Invalid field path for {backend}: component '{component}' too long ({len(component)} chars, max 64)"
            )

    return ".".join(field_path)
