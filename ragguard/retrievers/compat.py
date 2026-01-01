"""
Backward compatibility utilities for retriever API standardization.

This module provides utilities to help with parameter name transitions
while maintaining backward compatibility.
"""

import warnings
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


# Standard parameter names across all retrievers
STANDARD_PARAMS = {
    # Collection/container name
    "collection_name": ["collection", "index", "table"],
    # Vector embedding column
    "vector_field": ["embedding_column", "embedding_field"],
    # Database client
    "client": ["connection", "driver", "database"],
}

# Reverse mapping for lookup
DEPRECATED_TO_STANDARD: Dict[str, str] = {}
for standard, deprecated_list in STANDARD_PARAMS.items():
    for deprecated in deprecated_list:
        DEPRECATED_TO_STANDARD[deprecated] = standard


def deprecate_param(
    old_name: str,
    new_name: str,
    version: str = "1.0.0",
) -> Callable[[F], F]:
    """
    Decorator to add deprecation warning for renamed parameters.

    Usage:
        @deprecate_param("collection", "collection_name")
        def __init__(self, collection_name: str, ...):
            ...

    This allows both old and new parameter names to work, with a warning
    for the old name.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if old_name in kwargs:
                warnings.warn(
                    f"Parameter '{old_name}' is deprecated and will be removed in "
                    f"version {version}. Use '{new_name}' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                if new_name not in kwargs:
                    kwargs[new_name] = kwargs.pop(old_name)
                else:
                    # Both specified - use new name, ignore old
                    kwargs.pop(old_name)
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def standardize_kwargs(
    kwargs: Dict[str, Any],
    param_mappings: Optional[Dict[str, str]] = None,
    warn: bool = True,
    version: str = "1.0.0",
) -> Dict[str, Any]:
    """
    Standardize keyword arguments by mapping old names to new names.

    Args:
        kwargs: Original keyword arguments
        param_mappings: Mapping of old_name -> new_name (uses defaults if None)
        warn: Whether to emit deprecation warnings
        version: Version when old names will be removed

    Returns:
        Updated kwargs with standardized names

    Example:
        kwargs = standardize_kwargs(
            {"collection": "my_coll"},
            {"collection": "collection_name"}
        )
        # Returns {"collection_name": "my_coll"}
    """
    if param_mappings is None:
        param_mappings = DEPRECATED_TO_STANDARD

    result = dict(kwargs)

    for old_name, new_name in param_mappings.items():
        if old_name in result:
            if warn:
                warnings.warn(
                    f"Parameter '{old_name}' is deprecated and will be removed in "
                    f"version {version}. Use '{new_name}' instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )
            if new_name not in result:
                result[new_name] = result.pop(old_name)
            else:
                # Both specified - use new name, ignore old
                result.pop(old_name)

    return result


class ParameterAlias:
    """
    Descriptor that provides backward-compatible parameter aliases.

    Usage:
        class MyRetriever:
            # Old name is 'collection', new name is 'collection_name'
            collection = ParameterAlias("collection_name")

            def __init__(self, collection_name: str):
                self._collection_name = collection_name

            @property
            def collection_name(self):
                return self._collection_name
    """

    def __init__(self, target_name: str, version: str = "1.0.0"):
        self.target_name = target_name
        self.version = version
        self.attr_name: Optional[str] = None

    def __set_name__(self, owner, name):
        self.attr_name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        warnings.warn(
            f"Attribute '{self.attr_name}' is deprecated. "
            f"Use '{self.target_name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(obj, self.target_name)

    def __set__(self, obj, value):
        warnings.warn(
            f"Attribute '{self.attr_name}' is deprecated. "
            f"Use '{self.target_name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        setattr(obj, self.target_name, value)
