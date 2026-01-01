"""
Data models for condition compilation.

Defines the core enums and dataclasses used to represent compiled conditions,
values, and expressions in the policy engine.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple, Union


class ConditionOperator(Enum):
    """Supported operators in policy conditions."""
    EQUALS = "=="
    NOT_EQUALS = "!="
    IN = "in"
    NOT_IN = "not in"
    EXISTS = "exists"
    NOT_EXISTS = "not exists"
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN_OR_EQUAL = "<="


class ValueType(Enum):
    """Types of values that can appear in conditions."""
    USER_FIELD = "user_field"           # user.department
    DOCUMENT_FIELD = "document_field"   # document.department
    LITERAL_STRING = "literal_string"   # "engineering"
    LITERAL_NUMBER = "literal_number"   # 42 or 3.14
    LITERAL_BOOL = "literal_bool"       # true/false
    LITERAL_NONE = "literal_none"       # None
    LITERAL_LIST = "literal_list"       # ['a', 'b', 'c']


class ConditionType(Enum):
    """Types of condition patterns."""
    FIELD_IN_LIST = "field_in_list"         # document.field in ['a', 'b']
    VALUE_IN_ARRAY_FIELD = "value_in_array_field"  # user.id in document.authorized_users


class LogicalOperator(Enum):
    """Logical operators for combining conditions."""
    AND = "AND"
    OR = "OR"


@dataclass
class CompiledValue:
    """
    A compiled value expression.

    Represents the left or right side of a condition after parsing.
    Pre-splits field paths for efficient runtime access.
    """
    value_type: ValueType
    value: Any
    field_path: Tuple[str, ...]  # Pre-split path like ("metadata", "team")

    def __repr__(self) -> str:
        if self.field_path:
            path_str = ".".join(self.field_path)
            return f"CompiledValue({self.value_type.name}, {path_str})"
        return f"CompiledValue({self.value_type.name}, {self.value})"


@dataclass
class CompiledCondition:
    """
    A compiled condition expression.

    Pre-parsed representation of a condition like "user.dept == document.dept".
    This avoids string parsing on every evaluation.
    """
    operator: ConditionOperator
    left: CompiledValue
    right: CompiledValue
    original: str  # Original string for debugging
    condition_type: Optional['ConditionType'] = None  # Type of condition pattern

    def __repr__(self) -> str:
        return f"CompiledCondition({self.operator.name}, {self.left}, {self.right})"


@dataclass
class CompiledExpression:
    """
    Represents a logical expression tree for OR/AND operations.

    This enables complex boolean logic like:
    - (user.role == 'admin' OR user.role == 'manager')
    - ((A OR B) AND C)

    The expression tree consists of nodes that are either:
    - Individual conditions (CompiledCondition)
    - Sub-expressions (CompiledExpression)
    """
    operator: LogicalOperator
    children: list[Union[CompiledCondition, 'CompiledExpression']]
    original: str  # Original string for debugging

    def __repr__(self) -> str:
        return f"CompiledExpression({self.operator.name}, {len(self.children)} children)"

    def count_conditions(self) -> int:
        """
        Count total number of leaf conditions in this expression tree.

        Returns:
            Total number of CompiledCondition nodes
        """
        count = 0
        for child in self.children:
            if isinstance(child, CompiledCondition):
                count += 1
            elif isinstance(child, CompiledExpression):
                count += child.count_conditions()
        return count

    def get_depth(self) -> int:
        """
        Get the maximum nesting depth of this expression tree.

        Returns:
            Maximum nesting depth (0 for single level)

        Raises:
            RecursionError: If expression tree is too deeply nested
        """
        return self._get_depth_recursive(current_depth=0)

    def _get_depth_recursive(self, current_depth: int = 0) -> int:
        """
        Internal recursive method with depth protection.

        Args:
            current_depth: Current recursion depth

        Returns:
            Maximum nesting depth

        Raises:
            RecursionError: If recursion depth exceeds safe limits
        """
        # Protect against stack overflow by limiting recursion depth
        # This should never trigger with properly compiled expressions (checked during compilation)
        # but provides defense in depth if malformed data is present
        MAX_RECURSION_DEPTH = 100
        if current_depth >= MAX_RECURSION_DEPTH:
            raise RecursionError(
                f"Expression tree too deeply nested (depth >= {MAX_RECURSION_DEPTH}). "
                "This may indicate corrupted data or a bug in expression compilation."
            )

        max_child_depth = 0
        for child in self.children:
            if isinstance(child, CompiledExpression):
                child_depth = child._get_depth_recursive(current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
        return max_child_depth + 1
