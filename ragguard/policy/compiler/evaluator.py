"""
Condition evaluator for compiled conditions.

Efficiently evaluates compiled conditions and expressions against runtime
user and document contexts.

Security Note:
    This evaluator uses constant-time comparison for equality checks to
    prevent timing attacks. An attacker could otherwise measure response
    times to deduce information about document metadata or user attributes.
"""

import logging
from typing import Any, Tuple, Union

from ...utils import secure_compare, secure_contains
from .models import (
    CompiledCondition,
    CompiledExpression,
    CompiledValue,
    ConditionOperator,
    ConditionType,
    LogicalOperator,
    ValueType,
)

logger = logging.getLogger(__name__)


class CompiledConditionEvaluator:
    """
    Evaluates compiled conditions and expressions efficiently.

    This replaces the string-parsing evaluation logic in PolicyEngine
    with optimized compiled condition evaluation.

    Supports both simple conditions and complex boolean expressions with OR/AND.
    """

    @staticmethod
    def evaluate_node(
        node: Union[CompiledCondition, CompiledExpression],
        user: dict[str, Any],
        document: dict[str, Any]
    ) -> bool:
        """
        Evaluate a node (condition or expression).

        This is the main entry point that handles both:
        - Simple conditions (CompiledCondition)
        - Complex expressions (CompiledExpression)

        Args:
            node: Compiled condition or expression
            user: User context
            document: Document metadata

        Returns:
            True if condition/expression is satisfied, False otherwise
        """
        if isinstance(node, CompiledCondition):
            return CompiledConditionEvaluator.evaluate(node, user, document)
        elif isinstance(node, CompiledExpression):
            return CompiledConditionEvaluator.evaluate_expression(node, user, document)
        else:
            raise ValueError(f"Unknown node type: {type(node)}")

    @staticmethod
    def evaluate_expression(
        expr: CompiledExpression,
        user: dict[str, Any],
        document: dict[str, Any]
    ) -> bool:
        """
        Evaluate a compiled expression tree.

        Args:
            expr: Pre-compiled expression
            user: User context
            document: Document metadata

        Returns:
            True if expression is satisfied, False otherwise
        """
        if expr.operator == LogicalOperator.AND:
            # AND: all children must be true
            return all(
                CompiledConditionEvaluator.evaluate_node(child, user, document)
                for child in expr.children
            )
        elif expr.operator == LogicalOperator.OR:
            # OR: at least one child must be true
            return any(
                CompiledConditionEvaluator.evaluate_node(child, user, document)
                for child in expr.children
            )
        else:
            raise ValueError(f"Unknown logical operator: {expr.operator}")

    @staticmethod
    def evaluate(
        condition: CompiledCondition,
        user: dict[str, Any],
        document: dict[str, Any]
    ) -> bool:
        """
        Evaluate a compiled condition.

        Args:
            condition: Pre-compiled condition
            user: User context
            document: Document metadata

        Returns:
            True if condition is satisfied, False otherwise
        """
        # Resolve values
        left_value = CompiledConditionEvaluator._resolve_value(condition.left, user, document)
        right_value = CompiledConditionEvaluator._resolve_value(condition.right, user, document) if condition.right else None

        # Apply operator
        if condition.operator == ConditionOperator.EXISTS:
            # Check if field exists (is not None)
            return left_value is not None

        elif condition.operator == ConditionOperator.NOT_EXISTS:
            # Check if field does not exist (is None)
            return left_value is None

        elif condition.operator == ConditionOperator.GREATER_THAN:
            # Check if left > right
            if left_value is None or right_value is None:
                return False
            try:
                return left_value > right_value
            except TypeError:
                logger.warning(
                    "Type mismatch in comparison: cannot compare %s > %s (types: %s > %s)",
                    left_value, right_value, type(left_value).__name__, type(right_value).__name__,
                    exc_info=False
                )
                return False

        elif condition.operator == ConditionOperator.LESS_THAN:
            # Check if left < right
            if left_value is None or right_value is None:
                return False
            try:
                return left_value < right_value
            except TypeError:
                logger.warning(
                    "Type mismatch in comparison: cannot compare %s < %s (types: %s < %s)",
                    left_value, right_value, type(left_value).__name__, type(right_value).__name__,
                    exc_info=False
                )
                return False

        elif condition.operator == ConditionOperator.GREATER_THAN_OR_EQUAL:
            # Check if left >= right
            if left_value is None or right_value is None:
                return False
            try:
                return left_value >= right_value
            except TypeError:
                logger.warning(
                    "Type mismatch in comparison: cannot compare %s >= %s (types: %s >= %s)",
                    left_value, right_value, type(left_value).__name__, type(right_value).__name__,
                    exc_info=False
                )
                return False

        elif condition.operator == ConditionOperator.LESS_THAN_OR_EQUAL:
            # Check if left <= right
            if left_value is None or right_value is None:
                return False
            try:
                return left_value <= right_value
            except TypeError:
                logger.warning(
                    "Type mismatch in comparison: cannot compare %s <= %s (types: %s <= %s)",
                    left_value, right_value, type(left_value).__name__, type(right_value).__name__,
                    exc_info=False
                )
                return False

        elif condition.operator == ConditionOperator.EQUALS:
            # Security: Don't allow None == None to grant access
            if left_value is None or right_value is None:
                return False
            # Use constant-time comparison to prevent timing attacks
            return secure_compare(left_value, right_value)

        elif condition.operator == ConditionOperator.NOT_EQUALS:
            # Security: Missing fields should deny access, not grant it
            # If either value is None, return False (deny access)
            # Use EXISTS() operator if you want to explicitly allow missing fields
            if left_value is None or right_value is None:
                return False
            # Use constant-time comparison to prevent timing attacks
            return not secure_compare(left_value, right_value)

        elif condition.operator == ConditionOperator.IN:
            # Check condition type to determine evaluation strategy
            if condition.condition_type == ConditionType.VALUE_IN_ARRAY_FIELD:
                # Value IN array field: left_value in document's array field
                if not isinstance(right_value, list):
                    return False
                # Use constant-time comparison to prevent timing attacks
                return secure_contains(left_value, right_value)
            else:
                # Field IN literal list (normal case)
                if not isinstance(right_value, list):
                    return False
                # Use constant-time comparison to prevent timing attacks
                return secure_contains(left_value, right_value)

        elif condition.operator == ConditionOperator.NOT_IN:
            # Check condition type to determine evaluation strategy
            if condition.condition_type == ConditionType.VALUE_IN_ARRAY_FIELD:
                # Value NOT IN array field: left_value not in document's array field
                if not isinstance(right_value, list):
                    return False
                # Use constant-time comparison to prevent timing attacks
                return not secure_contains(left_value, right_value)
            else:
                # Field NOT IN literal list (normal case)
                if not isinstance(right_value, list):
                    return False
                # Use constant-time comparison to prevent timing attacks
                return not secure_contains(left_value, right_value)

        else:
            # Should never happen if compilation is correct
            raise ValueError(f"Unknown operator: {condition.operator}")

    @staticmethod
    def _resolve_value(
        compiled_value: CompiledValue,
        user: dict[str, Any],
        document: dict[str, Any]
    ) -> Any:
        """
        Resolve a compiled value to its actual runtime value.

        Args:
            compiled_value: Pre-compiled value
            user: User context
            document: Document metadata

        Returns:
            The resolved value
        """
        # Literal values - already resolved
        if compiled_value.value_type in (
            ValueType.LITERAL_STRING,
            ValueType.LITERAL_NUMBER,
            ValueType.LITERAL_BOOL,
            ValueType.LITERAL_NONE,
            ValueType.LITERAL_LIST
        ):
            return compiled_value.value

        # User field - resolve from user context using pre-split path
        if compiled_value.value_type == ValueType.USER_FIELD:
            return CompiledConditionEvaluator._get_nested_value(
                user,
                compiled_value.field_path
            )

        # Document field - resolve from document using pre-split path
        if compiled_value.value_type == ValueType.DOCUMENT_FIELD:
            return CompiledConditionEvaluator._get_nested_value(
                document,
                compiled_value.field_path
            )

        # Should never happen
        raise ValueError(f"Unknown value type: {compiled_value.value_type}")

    @staticmethod
    def _get_nested_value(obj: dict[str, Any], path: Tuple[str, ...]) -> Any:
        """
        Get a nested value using a pre-split path.

        This is faster than string splitting on every access.

        Args:
            obj: Dictionary to extract from
            path: Pre-split field path like ("metadata", "team")

        Returns:
            The nested value, or None if not found
        """
        value = obj
        for key in path:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return None
            else:
                return None
        return value
