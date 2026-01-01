"""
TigerGraph filter builder.

Converts RAGGuard policies to GSQL WHERE conditions for TigerGraph.
"""

from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Union

from ...exceptions import UnsupportedConditionError
from ...policy.models import Policy, Rule
from ..base import (
    get_nested_value,
    logger,
    parse_list_literal,
    parse_literal_value,
    user_satisfies_allow,
    validate_field_name,
)

if TYPE_CHECKING:
    from ...policy.compiler import CompiledCondition, CompiledExpression


def to_tigergraph_filter(
    policy: Policy,
    user: dict[str, Any],
    vertex_alias: str = "v"
) -> Tuple[str, dict[str, Any]]:
    """
    Convert a policy and user context to a GSQL WHERE clause.

    Args:
        policy: The access control policy
        user: User context
        vertex_alias: Alias for the vertex in the query (default: "v")

    Returns:
        Tuple of (where_clause, parameters)
        - where_clause: GSQL WHERE clause (without "WHERE" keyword)
        - parameters: Parameter dict for query binding

    Example:
        >>> where_clause, params = to_tigergraph_filter(policy, user)
        >>> query = f'''
        ...     SELECT v FROM Document:v
        ...     WHERE {where_clause}
        ...     LIMIT 10
        ... '''
    """
    conditions = []
    params = {}
    param_counter = [0]

    def next_param(prefix: str = "p") -> str:
        param_counter[0] += 1
        return f"{prefix}_{param_counter[0]}"

    for rule in policy.rules:
        rule_condition, rule_params = _build_tigergraph_rule_filter(
            rule, user, vertex_alias, next_param
        )
        if rule_condition is not None:
            conditions.append(rule_condition)
            params.update(rule_params)

    if not conditions:
        if policy.default == "allow":
            return "TRUE", {}
        else:
            return "FALSE", {}

    if len(conditions) == 1:
        return conditions[0], params

    # Multiple rules: OR them together
    combined = " OR ".join(f"({c})" for c in conditions)
    return combined, params


def _build_tigergraph_rule_filter(
    rule: Rule,
    user: dict[str, Any],
    vertex_alias: str,
    next_param: Callable[[str], str]
) -> Tuple[Optional[str], dict[str, Any]]:
    """
    Build a GSQL WHERE clause for a single rule.
    """
    if not user_satisfies_allow(rule.allow, user):
        return None, {}

    conditions = []
    params = {}

    # Add match conditions
    if rule.match:
        for key, value in rule.match.items():
            safe_key = validate_field_name(key, "tigergraph")

            if isinstance(value, list):
                # Use IN for list values
                param_name = next_param("match")
                # TigerGraph uses SetAccum for IN
                values_str = ", ".join(
                    f'"{v}"' if isinstance(v, str) else str(v)
                    for v in value
                )
                conditions.append(f"{vertex_alias}.{safe_key} IN ({values_str})")
            else:
                if isinstance(value, str):
                    conditions.append(f'{vertex_alias}.{safe_key} == "{value}"')
                else:
                    conditions.append(f"{vertex_alias}.{safe_key} == {value}")

    # Add conditions from policy
    if rule.allow.conditions:
        for condition in rule.allow.conditions:
            cond_str, cond_params = _build_tigergraph_condition_filter(
                condition, user, vertex_alias, next_param
            )
            if cond_str:
                conditions.append(cond_str)
                params.update(cond_params)

    if not conditions:
        return "TRUE", {}

    if len(conditions) == 1:
        return conditions[0], params

    combined = " AND ".join(conditions)
    return combined, params


def _build_tigergraph_from_compiled_node(
    node: Union['CompiledCondition', 'CompiledExpression'],
    user: dict[str, Any],
    vertex_alias: str,
    next_param: Callable[[str], str]
) -> Tuple[Optional[str], dict[str, Any]]:
    """
    Build a GSQL condition from a compiled expression node.
    """
    from ...policy.compiler import CompiledCondition, CompiledExpression, LogicalOperator

    if isinstance(node, CompiledCondition):
        return _build_tigergraph_from_condition(node, user, vertex_alias, next_param)

    elif isinstance(node, CompiledExpression):
        child_conditions = []
        params = {}

        for child in node.children:
            child_cond, child_params = _build_tigergraph_from_compiled_node(
                child, user, vertex_alias, next_param
            )
            if child_cond is not None:
                child_conditions.append(child_cond)
                params.update(child_params)

        if not child_conditions:
            return None, {}

        if len(child_conditions) == 1:
            return child_conditions[0], params

        if node.operator == LogicalOperator.OR:
            combined = " OR ".join(f"({c})" for c in child_conditions)
        else:
            combined = " AND ".join(child_conditions)

        return combined, params

    return None, {}


def _build_tigergraph_from_condition(
    condition: 'CompiledCondition',
    user: dict[str, Any],
    vertex_alias: str,
    next_param: Callable[[str], str]
) -> Tuple[Optional[str], dict[str, Any]]:
    """
    Build a GSQL condition from a CompiledCondition.
    """
    from ...policy.compiler import CompiledConditionEvaluator, ConditionOperator, ValueType

    # Resolve values
    left_value = None
    if condition.left.value_type == ValueType.USER_FIELD:
        left_value = CompiledConditionEvaluator._get_nested_value(
            user, condition.left.field_path
        )

    right_value = None
    if condition.right:
        if condition.right.value_type == ValueType.USER_FIELD:
            right_value = CompiledConditionEvaluator._get_nested_value(
                user, condition.right.field_path
            )
        elif condition.right.value_type in (
            ValueType.LITERAL_STRING,
            ValueType.LITERAL_NUMBER,
            ValueType.LITERAL_BOOL,
            ValueType.LITERAL_LIST
        ):
            right_value = condition.right.value
        elif condition.right.value_type == ValueType.LITERAL_NONE:
            right_value = None

    # Get document field
    doc_field = None
    if condition.left.value_type == ValueType.DOCUMENT_FIELD:
        doc_field = ".".join(condition.left.field_path)
        doc_field = validate_field_name(doc_field, "tigergraph")
    elif condition.right and condition.right.value_type == ValueType.DOCUMENT_FIELD:
        doc_field = ".".join(condition.right.field_path)
        doc_field = validate_field_name(doc_field, "tigergraph")

    def format_value(val):
        """Format a value for GSQL."""
        if isinstance(val, str):
            return f'"{val}"'
        elif isinstance(val, bool):
            return "TRUE" if val else "FALSE"
        elif val is None:
            return "NULL"
        return str(val)

    # Handle EXISTS
    if condition.operator == ConditionOperator.EXISTS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            # TigerGraph doesn't have direct EXISTS, check for non-empty
            return f"{vertex_alias}.{doc_field} != \"\"", {}

    # Handle NOT_EXISTS
    elif condition.operator == ConditionOperator.NOT_EXISTS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            return f"{vertex_alias}.{doc_field} == \"\"", {}

    # Handle EQUALS
    elif condition.operator == ConditionOperator.EQUALS:
        if (condition.left.value_type == ValueType.USER_FIELD and
                condition.right.value_type == ValueType.DOCUMENT_FIELD):
            if left_value is None:
                return "FALSE", {}
            return f"{vertex_alias}.{doc_field} == {format_value(left_value)}", {}

        elif condition.left.value_type == ValueType.DOCUMENT_FIELD:
            return f"{vertex_alias}.{doc_field} == {format_value(right_value)}", {}

    # Handle NOT_EQUALS
    elif condition.operator == ConditionOperator.NOT_EQUALS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            return f"{vertex_alias}.{doc_field} != {format_value(right_value)}", {}

    # Handle IN
    elif condition.operator == ConditionOperator.IN:
        if (condition.left.value_type == ValueType.USER_FIELD and
                condition.right.value_type == ValueType.DOCUMENT_FIELD):
            if left_value is None:
                return "FALSE", {}
            # Check if user value is in document's list attribute
            return f"{format_value(left_value)} IN {vertex_alias}.{doc_field}", {}

        elif (condition.left.value_type == ValueType.DOCUMENT_FIELD and
              right_value is not None and isinstance(right_value, list)):
            values_str = ", ".join(format_value(v) for v in right_value)
            return f"{vertex_alias}.{doc_field} IN ({values_str})", {}

    # Handle NOT_IN
    elif condition.operator == ConditionOperator.NOT_IN:
        if (condition.left.value_type == ValueType.USER_FIELD and
                condition.right.value_type == ValueType.DOCUMENT_FIELD):
            if left_value is None:
                return "TRUE", {}
            return f"NOT ({format_value(left_value)} IN {vertex_alias}.{doc_field})", {}

        elif (condition.left.value_type == ValueType.DOCUMENT_FIELD and
              right_value is not None and isinstance(right_value, list)):
            values_str = ", ".join(format_value(v) for v in right_value)
            return f"NOT ({vertex_alias}.{doc_field} IN ({values_str}))", {}

    # Handle comparison operators
    elif condition.operator == ConditionOperator.GREATER_THAN:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            return f"{vertex_alias}.{doc_field} > {format_value(right_value)}", {}

    elif condition.operator == ConditionOperator.LESS_THAN:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            return f"{vertex_alias}.{doc_field} < {format_value(right_value)}", {}

    elif condition.operator == ConditionOperator.GREATER_THAN_OR_EQUAL:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            return f"{vertex_alias}.{doc_field} >= {format_value(right_value)}", {}

    elif condition.operator == ConditionOperator.LESS_THAN_OR_EQUAL:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            return f"{vertex_alias}.{doc_field} <= {format_value(right_value)}", {}

    # If we reach here with an unhandled operator, raise an error instead of silently allowing
    raise UnsupportedConditionError(
        condition=str(condition.operator),
        backend="tigergraph",
        reason=f"Operator {condition.operator} with value types "
               f"left={condition.left.value_type}, right={condition.right.value_type if condition.right else None} "
               f"is not supported. This is a security measure to prevent silent filter bypass."
    )


def _build_tigergraph_condition_filter(
    condition: str,
    user: dict[str, Any],
    vertex_alias: str,
    next_param: Callable[[str], str]
) -> Tuple[Optional[str], dict[str, Any]]:
    """
    Build a GSQL condition from a condition expression.
    """
    from ...policy.compiler import CompiledExpression, ConditionCompiler

    try:
        compiled = ConditionCompiler.compile_expression(condition)

        if isinstance(compiled, CompiledExpression):
            return _build_tigergraph_from_compiled_node(
                compiled, user, vertex_alias, next_param
            )
        else:
            return _build_tigergraph_from_condition(
                compiled, user, vertex_alias, next_param
            )
    except (ValueError, AttributeError, KeyError, TypeError) as e:
        logger.debug(
            "Condition compilation failed, falling back to string parsing: %s (condition: %s)",
            str(e), condition
        )

    return _parse_condition_string(condition, user, vertex_alias, next_param)


def _parse_condition_string(
    condition: str,
    user: dict[str, Any],
    vertex_alias: str,
    next_param: Callable[[str], str]
) -> Tuple[Optional[str], dict[str, Any]]:
    """
    Parse a condition string and convert to GSQL.
    """
    condition = condition.strip()

    def format_value(val):
        if isinstance(val, str):
            return f'"{val}"'
        elif isinstance(val, bool):
            return "TRUE" if val else "FALSE"
        elif val is None:
            return "NULL"
        return str(val)

    # Parse field existence
    if " not exists" in condition:
        field = condition.replace(" not exists", "").strip()
        if field.startswith("document."):
            doc_field = validate_field_name(field[9:], "tigergraph")
            return f'{vertex_alias}.{doc_field} == ""', {}
        return None, {}

    elif " exists" in condition:
        field = condition.replace(" exists", "").strip()
        if field.startswith("document."):
            doc_field = validate_field_name(field[9:], "tigergraph")
            return f'{vertex_alias}.{doc_field} != ""', {}
        return None, {}

    # Parse !=
    elif "!=" in condition:
        parts = condition.split("!=", 1)
        if len(parts) != 2:
            return None, {}

        left, right = parts[0].strip(), parts[1].strip()

        if left.startswith("document."):
            doc_field = validate_field_name(left[9:], "tigergraph")
            literal_value = parse_literal_value(right)
            return f"{vertex_alias}.{doc_field} != {format_value(literal_value)}", {}

    # Parse ==
    elif "==" in condition:
        parts = condition.split("==", 1)
        if len(parts) != 2:
            return None, {}

        left, right = parts[0].strip(), parts[1].strip()

        if left.startswith("user."):
            user_field = left[5:]
            user_value = get_nested_value(user, user_field)

            if user_value is None:
                return "FALSE", {}

            if right.startswith("document."):
                doc_field = validate_field_name(right[9:], "tigergraph")
                return f"{vertex_alias}.{doc_field} == {format_value(user_value)}", {}

        elif left.startswith("document."):
            doc_field = validate_field_name(left[9:], "tigergraph")
            literal_value = parse_literal_value(right)
            return f"{vertex_alias}.{doc_field} == {format_value(literal_value)}", {}

    # Parse "not in"
    elif " not in " in condition:
        parts = condition.split(" not in ", 1)
        if len(parts) != 2:
            return None, {}

        left, right = parts[0].strip(), parts[1].strip()

        if left.startswith("user.") and right.startswith("document."):
            user_field = left[5:]
            doc_field = validate_field_name(right[9:], "tigergraph")
            user_value = get_nested_value(user, user_field)

            if user_value is None:
                return "TRUE", {}

            return f"NOT ({format_value(user_value)} IN {vertex_alias}.{doc_field})", {}

        elif left.startswith("document."):
            doc_field = validate_field_name(left[9:], "tigergraph")
            list_values = parse_list_literal(right)

            if list_values is not None and len(list_values) > 0:
                values_str = ", ".join(format_value(v) for v in list_values)
                return f"NOT ({vertex_alias}.{doc_field} IN ({values_str}))", {}

    # Parse "in"
    elif " in " in condition:
        parts = condition.split(" in ", 1)
        if len(parts) != 2:
            return None, {}

        left, right = parts[0].strip(), parts[1].strip()

        if left.startswith("user.") and right.startswith("document."):
            user_field = left[5:]
            doc_field = validate_field_name(right[9:], "tigergraph")
            user_value = get_nested_value(user, user_field)

            if user_value is None:
                return "FALSE", {}

            return f"{format_value(user_value)} IN {vertex_alias}.{doc_field}", {}

        elif left.startswith("document."):
            doc_field = validate_field_name(left[9:], "tigergraph")
            list_values = parse_list_literal(right)

            if list_values is not None:
                if len(list_values) == 0:
                    return "FALSE", {}
                values_str = ", ".join(format_value(v) for v in list_values)
                return f"{vertex_alias}.{doc_field} IN ({values_str})", {}

    return None, {}
