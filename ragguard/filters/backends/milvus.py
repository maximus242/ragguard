"""
Milvus filter builder.

Converts RAGGuard policies to Milvus-native filter expressions.
"""

from typing import TYPE_CHECKING, Any, Optional, Union

from ...exceptions import UnsupportedConditionError
from ...policy.models import Policy, Rule
from ..base import (
    SKIP_RULE,
    get_nested_value,
    parse_literal_value,
    user_satisfies_allow,
    validate_field_name,
    validate_field_path,
)

if TYPE_CHECKING:
    from ...policy.compiler import CompiledCondition, CompiledExpression


def to_milvus_filter(policy: Policy, user: dict[str, Any]) -> Optional[str]:
    """
    Convert a policy and user context to a Milvus filter expression.

    Milvus uses a SQL-like expression language for filtering.
    This function generates native Milvus filter expressions for optimal performance.

    Args:
        policy: The access control policy
        user: User context

    Returns:
        Milvus filter expression string, or None if no filter needed

    Example:
        Input: user.department == document.department
        Output: "department == 'engineering'"

        Input: (user.dept == doc.dept OR doc.visibility == 'public')
        Output: "(department == 'engineering' or visibility == 'public')"

    Milvus Expression Language:
        - Operators: ==, !=, >, >=, <, <=
        - Logical: and, or, not
        - Membership: in, not in
        - Pattern: like (SQL LIKE)
        - Examples:
          * "department == 'engineering'"
          * "age >= 18 and city in ['NYC', 'LA']"
          * "name like 'John%'"
    """
    or_clauses = []  # OR between rules

    for rule in policy.rules:
        rule_expr = _build_milvus_rule_filter(rule, user)
        if rule_expr is SKIP_RULE:
            # User doesn't satisfy this rule's allow conditions - skip to next
            continue
        if rule_expr is None:
            # This rule matches all documents - return None (no filter needed)
            return None
        or_clauses.append(rule_expr)

    if not or_clauses:
        # No rules apply to this user
        if policy.default == "allow":
            return None  # No filter needed (allow all)
        else:
            return "id < 0"  # Deny all (impossible condition)

    # Combine with OR
    if len(or_clauses) == 1:
        return or_clauses[0]
    else:
        return "(" + " or ".join(f"({clause})" for clause in or_clauses) + ")"


def _build_milvus_rule_filter(rule: Rule, user: dict[str, Any]) -> Optional[str]:
    """Build Milvus filter expression for a single rule.

    Returns:
        - str: A filter expression for this rule
        - None: Rule matches all documents (no restrictions)
        - SKIP_RULE: User doesn't satisfy this rule's allow conditions
    """
    # Check if user satisfies the allow conditions
    if not user_satisfies_allow(rule.allow, user):
        return SKIP_RULE

    and_clauses = []

    # Add match conditions
    if rule.match:
        for key, value in rule.match.items():
            if isinstance(value, bool):
                and_clauses.append(f"{key} == {str(value).lower()}")
            elif isinstance(value, (int, float)):
                and_clauses.append(f"{key} == {value}")
            elif isinstance(value, str):
                and_clauses.append(f"{key} == '{_escape_milvus_string(value)}'")
            elif isinstance(value, list):
                # IN clause
                values_str = ", ".join([
                    f"'{_escape_milvus_string(v)}'" if isinstance(v, str) else str(v)
                    for v in value
                ])
                and_clauses.append(f"{key} in [{values_str}]")

    # Add conditions that reference user context
    if rule.allow.conditions:
        for condition in rule.allow.conditions:
            cond_expr = _build_milvus_condition_filter(condition, user)
            if cond_expr:
                and_clauses.append(cond_expr)

    if not and_clauses:
        # No conditions - matches all documents
        return None

    # Combine with AND
    if len(and_clauses) == 1:
        return and_clauses[0]
    else:
        return "(" + " and ".join(and_clauses) + ")"


def _build_milvus_condition_filter(
    condition: str,
    user: dict[str, Any]
) -> Optional[str]:
    """
    Build a Milvus expression from a policy condition.

    v0.3.0: Now supports native OR/AND logic by detecting CompiledExpression.
    """
    from ...policy.compiler import CompiledCondition, CompiledExpression, ConditionCompiler

    # v0.3.0: Try to compile as expression (supports OR/AND)
    try:
        compiled = ConditionCompiler.compile_expression(condition)

        # Use native Milvus expression builder for both CompiledExpression and CompiledCondition
        if isinstance(compiled, (CompiledExpression, CompiledCondition)):
            return _build_milvus_from_compiled_node(compiled, user)

    except (ValueError, AttributeError, KeyError, TypeError):
        # Only catch specific exceptions, not system exceptions
        pass

    # Fallback to string parsing (backward compatibility)
    condition = condition.strip()

    # Parse user.field == document.field
    if "==" in condition:
        parts = condition.split("==", 1)
        if len(parts) != 2:
            return None

        left, right = parts[0].strip(), parts[1].strip()

        if left.startswith("user.") and right.startswith("document."):
            user_field = left[5:]
            doc_field = right[9:]
            validate_field_name(doc_field, "milvus")
            user_value = get_nested_value(user, user_field)

            if user_value is None:
                return "id < 0"  # Deny (impossible condition)

            if isinstance(user_value, bool):
                return f"{doc_field} == {str(user_value).lower()}"
            elif isinstance(user_value, (int, float)):
                return f"{doc_field} == {user_value}"
            elif isinstance(user_value, str):
                return f"{doc_field} == '{_escape_milvus_string(user_value)}'"

        # document.field == 'literal'
        elif left.startswith("document."):
            doc_field = left[9:]
            validate_field_name(doc_field, "milvus")
            literal_value = parse_literal_value(right)

            if isinstance(literal_value, bool):
                return f"{doc_field} == {str(literal_value).lower()}"
            elif isinstance(literal_value, (int, float)):
                return f"{doc_field} == {literal_value}"
            elif isinstance(literal_value, str):
                return f"{doc_field} == '{_escape_milvus_string(literal_value)}'"

    # Parse user.field in document.array_field
    elif " in " in condition:
        parts = condition.split(" in ")
        if len(parts) != 2:
            return None

        left, right = parts[0].strip(), parts[1].strip()

        if left.startswith("user.") and right.startswith("document."):
            user_field = left[5:]
            doc_field = right[9:]
            validate_field_name(doc_field, "milvus")
            user_value = get_nested_value(user, user_field)

            if user_value is None:
                return "id < 0"  # Deny

            # Milvus: check if value is in array field using JSON contains
            # For now, use string matching (Milvus has limited array operations)
            if isinstance(user_value, str):
                return f"array_contains({doc_field}, '{_escape_milvus_string(user_value)}')"

    return None


def _build_milvus_from_compiled_node(
    node: Union['CompiledCondition', 'CompiledExpression'],
    user: dict[str, Any]
) -> Optional[str]:
    """
    Build Milvus expression from a compiled expression node.

    Enables native OR/AND support by converting expression trees to Milvus syntax.

    Returns:
        Milvus expression string
    """
    from ...policy.compiler import CompiledCondition, CompiledExpression, LogicalOperator

    if isinstance(node, CompiledCondition):
        return _build_milvus_from_condition(node, user)

    elif isinstance(node, CompiledExpression):
        child_exprs = []

        for child in node.children:
            expr = _build_milvus_from_compiled_node(child, user)
            if expr:
                child_exprs.append(expr)

        if not child_exprs:
            return None

        if len(child_exprs) == 1:
            return child_exprs[0]

        # Build native OR/AND expression
        if node.operator == LogicalOperator.OR:
            return "(" + " or ".join(f"({expr})" for expr in child_exprs) + ")"
        elif node.operator == LogicalOperator.AND:
            return "(" + " and ".join(f"({expr})" for expr in child_exprs) + ")"

    return None


def _build_milvus_from_condition(
    condition: 'CompiledCondition',
    user: dict[str, Any]
) -> Optional[str]:
    """
    Build Milvus expression from a CompiledCondition.

    Converts the compiled AST to Milvus expression syntax.
    """
    from ...policy.compiler import CompiledConditionEvaluator, ConditionOperator, ValueType

    # Resolve user values
    left_value = None
    if condition.left.value_type == ValueType.USER_FIELD:
        left_value = CompiledConditionEvaluator._get_nested_value(user, condition.left.field_path)

    right_value = None
    if condition.right:
        if condition.right.value_type == ValueType.USER_FIELD:
            right_value = CompiledConditionEvaluator._get_nested_value(user, condition.right.field_path)
        elif condition.right.value_type in (ValueType.LITERAL_STRING, ValueType.LITERAL_NUMBER, ValueType.LITERAL_BOOL, ValueType.LITERAL_LIST):
            right_value = condition.right.value
        elif condition.right.value_type == ValueType.LITERAL_NONE:
            right_value = None

    # Build expression based on operator
    if condition.operator == ConditionOperator.EQUALS:
        # user.field == document.field
        if condition.left.value_type == ValueType.USER_FIELD and condition.right.value_type == ValueType.DOCUMENT_FIELD:
            if left_value is None:
                return "id < 0"  # Deny
            doc_field = validate_field_path(condition.right.field_path, "milvus")
            return _milvus_equals_expr(doc_field, left_value)

        # document.field == literal
        elif condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = validate_field_path(condition.left.field_path, "milvus")
            return _milvus_equals_expr(doc_field, right_value)

    elif condition.operator == ConditionOperator.NOT_EQUALS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = validate_field_path(condition.left.field_path, "milvus")
            return _milvus_not_equals_expr(doc_field, right_value)

    elif condition.operator == ConditionOperator.IN:
        # user.id in document.array_field
        if condition.left.value_type == ValueType.USER_FIELD and condition.right.value_type == ValueType.DOCUMENT_FIELD:
            if left_value is None:
                return "id < 0"
            doc_field = validate_field_path(condition.right.field_path, "milvus")
            # Milvus array_contains function
            if isinstance(left_value, str):
                return f"array_contains({doc_field}, '{_escape_milvus_string(left_value)}')"
            else:
                return f"array_contains({doc_field}, {left_value})"

        # document.field in [literals]
        elif condition.left.value_type == ValueType.DOCUMENT_FIELD and isinstance(right_value, list):
            if not right_value:
                return "id < 0"
            doc_field = validate_field_path(condition.left.field_path, "milvus")
            values_str = ", ".join([
                f"'{_escape_milvus_string(v)}'" if isinstance(v, str) else str(v)
                for v in right_value
            ])
            return f"{doc_field} in [{values_str}]"

        # literal in document.array_field (e.g., 'public' in document.tags)
        elif condition.left.value_type == ValueType.LITERAL_STRING and condition.right.value_type == ValueType.DOCUMENT_FIELD:
            literal_value = condition.left.value
            doc_field = validate_field_path(condition.right.field_path, "milvus")
            if isinstance(literal_value, str):
                return f"array_contains({doc_field}, '{_escape_milvus_string(literal_value)}')"
            else:
                return f"array_contains({doc_field}, {literal_value})"

    elif condition.operator == ConditionOperator.GREATER_THAN:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = validate_field_path(condition.left.field_path, "milvus")
            return f"{doc_field} > {right_value}"

    elif condition.operator == ConditionOperator.LESS_THAN:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = validate_field_path(condition.left.field_path, "milvus")
            return f"{doc_field} < {right_value}"

    elif condition.operator == ConditionOperator.GREATER_THAN_OR_EQUAL:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = validate_field_path(condition.left.field_path, "milvus")
            return f"{doc_field} >= {right_value}"

    elif condition.operator == ConditionOperator.LESS_THAN_OR_EQUAL:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = validate_field_path(condition.left.field_path, "milvus")
            return f"{doc_field} <= {right_value}"

    # Handle EXISTS operator
    elif condition.operator == ConditionOperator.EXISTS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = validate_field_path(condition.left.field_path, "milvus")
            # Milvus: check field is not null/empty using JSON path exists
            # Note: Milvus 2.3+ supports JSON field existence checks
            return f"{doc_field} != ''"

    # Handle NOT_EXISTS operator
    elif condition.operator == ConditionOperator.NOT_EXISTS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = validate_field_path(condition.left.field_path, "milvus")
            # Milvus: check field is null/empty
            return f"{doc_field} == ''"

    # Handle NOT_IN operator
    elif condition.operator == ConditionOperator.NOT_IN:
        # user.id not in document.array_field
        if condition.left.value_type == ValueType.USER_FIELD and condition.right.value_type == ValueType.DOCUMENT_FIELD:
            if left_value is None:
                return None  # No restriction if user field is missing
            doc_field = validate_field_path(condition.right.field_path, "milvus")
            # Milvus: negate array_contains
            if isinstance(left_value, str):
                return f"not array_contains({doc_field}, '{_escape_milvus_string(left_value)}')"
            else:
                return f"not array_contains({doc_field}, {left_value})"

        # document.field not in [literals]
        elif condition.left.value_type == ValueType.DOCUMENT_FIELD and isinstance(right_value, list):
            if not right_value:
                return None  # Empty list means no restriction
            doc_field = validate_field_path(condition.left.field_path, "milvus")
            values_str = ", ".join([
                f"'{_escape_milvus_string(v)}'" if isinstance(v, str) else str(v)
                for v in right_value
            ])
            return f"{doc_field} not in [{values_str}]"

        # literal not in document.array_field (e.g., 'archived' not in document.tags)
        elif condition.left.value_type == ValueType.LITERAL_STRING and condition.right.value_type == ValueType.DOCUMENT_FIELD:
            literal_value = condition.left.value
            doc_field = validate_field_path(condition.right.field_path, "milvus")
            if isinstance(literal_value, str):
                return f"not array_contains({doc_field}, '{_escape_milvus_string(literal_value)}')"
            else:
                return f"not array_contains({doc_field}, {literal_value})"

    # If we reach here with an unhandled operator, raise an error instead of silently allowing
    # This prevents security bypass from unsupported operators
    raise UnsupportedConditionError(
        condition=str(condition.operator),
        backend="milvus",
        reason=f"Operator {condition.operator} with value types "
               f"left={condition.left.value_type}, right={condition.right.value_type if condition.right else None} "
               f"is not supported. This is a security measure to prevent silent filter bypass."
    )


def _milvus_equals_expr(field: str, value: Any) -> str:
    """Generate Milvus equals expression."""
    if isinstance(value, bool):
        return f"{field} == {str(value).lower()}"
    elif isinstance(value, (int, float)):
        return f"{field} == {value}"
    elif isinstance(value, str):
        return f"{field} == '{_escape_milvus_string(value)}'"
    elif value is None:
        return "id < 0"  # Milvus doesn't support null checks well
    else:
        return f"{field} == {value}"


def _milvus_not_equals_expr(field: str, value: Any) -> str:
    """Generate Milvus not-equals expression."""
    if isinstance(value, bool):
        return f"{field} != {str(value).lower()}"
    elif isinstance(value, (int, float)):
        return f"{field} != {value}"
    elif isinstance(value, str):
        return f"{field} != '{_escape_milvus_string(value)}'"
    else:
        return f"{field} != {value}"


def _escape_milvus_string(s: str) -> str:
    """Escape string for Milvus expression."""
    # Escape single quotes by doubling them (SQL-style)
    return s.replace("'", "''")
