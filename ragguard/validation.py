"""
Input validation and sanitization for RAGGuard.

Validates user and document contexts to prevent:
- DOS attacks via deeply nested objects or large payloads
- Field name injection attacks
- Type confusion attacks
- Invalid data causing crashes
"""

import re
from typing import Any, Dict, List, Optional

from .errors import empty_user_context_error
from .exceptions import RetrieverError

# Maximum sizes to prevent DOS attacks
DEFAULT_MAX_DICT_SIZE = 100  # Maximum number of fields
DEFAULT_MAX_STRING_LENGTH = 10000  # Maximum string length
DEFAULT_MAX_ARRAY_LENGTH = 1000  # Maximum array length
DEFAULT_MAX_NESTING_DEPTH = 10  # Maximum nesting depth


# Dangerous field name patterns
DANGEROUS_PATTERNS = [
    r"^__",  # Double underscore (Python internals)
    r"\$",  # Dollar sign (MongoDB injection)
    r"[;\s]",  # Semicolons or SQL commands
    r"<script",  # XSS attempts
    r"javascript:",  # JavaScript injection
    r"\x00",  # Null bytes
]

# Compiled regex for efficiency
DANGEROUS_PATTERN_RE = re.compile("|".join(DANGEROUS_PATTERNS), re.IGNORECASE)


class ValidationConfig:
    """
    Configuration for input validation.

    Allows customizing validation limits per use case.
    """

    def __init__(
        self,
        max_dict_size: int = DEFAULT_MAX_DICT_SIZE,
        max_string_length: int = DEFAULT_MAX_STRING_LENGTH,
        max_array_length: int = DEFAULT_MAX_ARRAY_LENGTH,
        max_nesting_depth: int = DEFAULT_MAX_NESTING_DEPTH,
        allow_none_values: bool = True,
        strict_field_names: bool = True,
    ):
        """
        Initialize validation configuration.

        Args:
            max_dict_size: Maximum number of fields in a dictionary
            max_string_length: Maximum string length
            max_array_length: Maximum array length
            max_nesting_depth: Maximum object nesting depth
            allow_none_values: Whether to allow None values
            strict_field_names: Whether to enforce strict field name validation
        """
        self.max_dict_size = max_dict_size
        self.max_string_length = max_string_length
        self.max_array_length = max_array_length
        self.max_nesting_depth = max_nesting_depth
        self.allow_none_values = allow_none_values
        self.strict_field_names = strict_field_names


class InputValidator:
    """
    Validates user and document contexts.

    Ensures inputs are safe and within acceptable bounds.
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize validator with configuration.

        Args:
            config: Validation configuration (uses defaults if not provided)
        """
        self.config = config or ValidationConfig()

    def validate_user_context(self, user: Dict[str, Any]) -> None:
        """
        Validate user context dictionary.

        Args:
            user: User context to validate

        Raises:
            RetrieverError: If validation fails
        """
        if not isinstance(user, dict):
            raise RetrieverError(
                f"User context must be a dictionary, got {type(user).__name__}"
            )

        if not user:
            raise RetrieverError(empty_user_context_error(require_id=True))

        self._validate_dict(user, "user", depth=0)

    def validate_document_context(self, document: Dict[str, Any]) -> None:
        """
        Validate document context dictionary.

        Args:
            document: Document metadata to validate

        Raises:
            RetrieverError: If validation fails
        """
        if not isinstance(document, dict):
            raise RetrieverError(
                f"Document context must be a dictionary, got {type(document).__name__}"
            )

        # Document can be empty (for "everyone" policies)
        if document:
            self._validate_dict(document, "document", depth=0)

    def _validate_dict(self, data: Dict[str, Any], context: str, depth: int) -> None:
        """
        Recursively validate a dictionary.

        Args:
            data: Dictionary to validate
            context: Context name for error messages
            depth: Current nesting depth

        Raises:
            RetrieverError: If validation fails
        """
        if depth > self.config.max_nesting_depth:
            raise RetrieverError(
                f"{context} exceeds maximum nesting depth of {self.config.max_nesting_depth}"
            )

        if len(data) > self.config.max_dict_size:
            raise RetrieverError(
                f"{context} has {len(data)} fields, maximum is {self.config.max_dict_size}"
            )

        for key, value in data.items():
            self._validate_field_name(key, context)
            self._validate_value(value, f"{context}.{key}", depth + 1)

    def _validate_field_name(self, name: str, context: str) -> None:
        """
        Validate field name for dangerous patterns.

        Args:
            name: Field name to validate
            context: Context name for error messages

        Raises:
            RetrieverError: If field name is invalid
        """
        if not isinstance(name, str):
            raise RetrieverError(
                f"{context} field name must be string, got {type(name).__name__}"
            )

        if not name:
            raise RetrieverError(f"{context} field name cannot be empty")

        if DANGEROUS_PATTERN_RE.search(name):
            raise RetrieverError(
                f"Invalid field name '{name}' in {context}: contains dangerous pattern"
            )

        if self.config.strict_field_names:
            if not re.match(r'^[a-zA-Z0-9_.\-]+$', name):
                raise RetrieverError(
                    f"Invalid field name '{name}' in {context}: "
                    f"must contain only letters, numbers, underscores, hyphens, and dots"
                )

    def _validate_value(self, value: Any, context: str, depth: int) -> None:
        """
        Validate a field value.

        Args:
            value: Value to validate
            context: Context name for error messages
            depth: Current nesting depth

        Raises:
            RetrieverError: If value is invalid
        """
        if value is None:
            if not self.config.allow_none_values:
                raise RetrieverError(f"{context} has None value but None values are not allowed")
            return

        if isinstance(value, str):
            self._validate_string(value, context)
        elif isinstance(value, (int, float, bool)):
            pass
        elif isinstance(value, list):
            self._validate_array(value, context, depth)
        elif isinstance(value, dict):
            self._validate_dict(value, context, depth)
        else:
            raise RetrieverError(
                f"{context} has unsupported type {type(value).__name__}. "
                f"Allowed types: str, int, float, bool, list, dict, None"
            )

    def _validate_string(self, value: str, context: str) -> None:
        """
        Validate string value.

        Args:
            value: String to validate
            context: Context name for error messages

        Raises:
            RetrieverError: If string is invalid
        """
        if len(value) > self.config.max_string_length:
            raise RetrieverError(
                f"{context} string length {len(value)} exceeds maximum {self.config.max_string_length}"
            )

        if '\x00' in value:
            raise RetrieverError(f"{context} contains null byte")

    def _validate_array(self, value: List[Any], context: str, depth: int) -> None:
        """
        Validate array value.

        Args:
            value: Array to validate
            context: Context name for error messages
            depth: Current nesting depth

        Raises:
            RetrieverError: If array is invalid
        """
        if len(value) > self.config.max_array_length:
            raise RetrieverError(
                f"{context} array length {len(value)} exceeds maximum {self.config.max_array_length}"
            )

        for i, item in enumerate(value):
            self._validate_value(item, f"{context}[{i}]", depth)


# Global default validator
_default_validator = InputValidator()


def validate_user(user: Dict[str, Any], config: Optional[ValidationConfig] = None) -> None:
    """
    Validate user context (convenience function).

    Args:
        user: User context to validate
        config: Optional validation configuration

    Raises:
        RetrieverError: If validation fails

    Example:
        >>> from ragguard.validation import validate_user
        >>> user = {"id": "alice", "department": "engineering"}
        >>> validate_user(user)  # OK
        >>>
        >>> bad_user = {"__proto__": "malicious"}
        >>> validate_user(bad_user)  # Raises RetrieverError
    """
    if config:
        validator = InputValidator(config)
        validator.validate_user_context(user)
    else:
        _default_validator.validate_user_context(user)


def validate_document(document: Dict[str, Any], config: Optional[ValidationConfig] = None) -> None:
    """
    Validate document context (convenience function).

    Args:
        document: Document metadata to validate
        config: Optional validation configuration

    Raises:
        RetrieverError: If validation fails

    Example:
        >>> from ragguard.validation import validate_document
        >>> doc = {"id": "doc1", "department": "engineering"}
        >>> validate_document(doc)  # OK
        >>>
        >>> bad_doc = {"field": "x" * 100000}
        >>> validate_document(bad_doc)  # Raises RetrieverError
    """
    if config:
        validator = InputValidator(config)
        validator.validate_document_context(document)
    else:
        _default_validator.validate_document_context(document)


__all__ = [
    "InputValidator",
    "ValidationConfig",
    "validate_document",
    "validate_user",
]
