"""
ArangoDB filter builder.

Converts RAGGuard policies to AQL FILTER conditions for ArangoDB.
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


def to_arangodb_filter(
    policy: Policy,
    user: dict[str, Any],
    doc_alias: str = "doc"
) -> Tuple[str, dict[str, Any]]:
    """
    Convert a policy and user context to an AQL FILTER clause.

    Args:
        policy: The access control policy
        user: User context
        doc_alias: Alias for the document in the query (default: "doc")

    Returns:
        Tuple of (filter_clause, bind_vars)
        - filter_clause: AQL FILTER clause (without "FILTER" keyword)
        - bind_vars: Bind variables for the query

    Example:
        >>> filter_clause, bind_vars = to_arangodb_filter(policy, user)
        >>> query = f'''
        ...     FOR doc IN documents
        ...     FILTER {filter_clause}
        ...     LIMIT 10
        ...     RETURN doc
        ... '''
        >>> db.aql.execute(query, bind_vars=bind_vars)
    """
    conditions = []
    bind_vars = {}
    param_counter = [0]

    def next_param(prefix: str = "p") -> str:
        param_counter[0] += 1
        return f"{prefix}_{param_counter[0]}"

    for rule in policy.rules:
        rule_condition, rule_vars = _build_arangodb_rule_filter(
            rule, user, doc_alias, next_param
        )
        if rule_condition is not None:
            conditions.append(rule_condition)
            bind_vars.update(rule_vars)

    if not conditions:
        if policy.default == "allow":
            return "true", {}
        else:
            return "false", {}

    if len(conditions) == 1:
        return conditions[0], bind_vars

    # Multiple rules: OR them together
    combined = " || ".join(f"({c})" for c in conditions)
    return combined, bind_vars


def _build_arangodb_rule_filter(
    rule: Rule,
    user: dict[str, Any],
    doc_alias: str,
    next_param: Callable[[str], str]
) -> Tuple[Optional[str], dict[str, Any]]:
    """
    Build an AQL FILTER condition for a single rule.
    """
    if not user_satisfies_allow(rule.allow, user):
        return None, {}

    conditions = []
    bind_vars = {}

    # Add match conditions
    if rule.match:
        for key, value in rule.match.items():
            safe_key = validate_field_name(key, "arangodb")

            if isinstance(value, list):
                param_name = next_param("match")
                bind_vars[param_name] = value
                conditions.append(f"{doc_alias}.{safe_key} IN @{param_name}")
            else:
                param_name = next_param("match")
                bind_vars[param_name] = value
                conditions.append(f"{doc_alias}.{safe_key} == @{param_name}")

    # Add conditions from policy
    if rule.allow.conditions:
        for condition in rule.allow.conditions:
            cond_str, cond_vars = _build_arangodb_condition_filter(
                condition, user, doc_alias, next_param
            )
            if cond_str:
                conditions.append(cond_str)
                bind_vars.update(cond_vars)

    if not conditions:
        return "true", {}

    if len(conditions) == 1:
        return conditions[0], bind_vars

    combined = " && ".join(conditions)
    return combined, bind_vars


def _build_arangodb_from_compiled_node(
    node: Union['CompiledCondition', 'CompiledExpression'],
    user: dict[str, Any],
    doc_alias: str,
    next_param: Callable[[str], str]
) -> Tuple[Optional[str], dict[str, Any]]:
    """
    Build an AQL condition from a compiled expression node.
    """
    from ...policy.compiler import CompiledCondition, CompiledExpression, LogicalOperator

    if isinstance(node, CompiledCondition):
        return _build_arangodb_from_condition(node, user, doc_alias, next_param)

    elif isinstance(node, CompiledExpression):
        child_conditions = []
        bind_vars = {}

        for child in node.children:
            child_cond, child_vars = _build_arangodb_from_compiled_node(
                child, user, doc_alias, next_param
            )
            if child_cond is not None:
                child_conditions.append(child_cond)
                bind_vars.update(child_vars)

        if not child_conditions:
            return None, {}

        if len(child_conditions) == 1:
            return child_conditions[0], bind_vars

        if node.operator == LogicalOperator.OR:
            combined = " || ".join(f"({c})" for c in child_conditions)
        else:
            combined = " && ".join(child_conditions)

        return combined, bind_vars

    return None, {}


def _build_arangodb_from_condition(
    condition: 'CompiledCondition',
    user: dict[str, Any],
    doc_alias: str,
    next_param: Callable[[str], str]
) -> Tuple[Optional[str], dict[str, Any]]:
    """
    Build an AQL condition from a CompiledCondition.
    """
    from ...policy.compiler import CompiledConditionEvaluator, ConditionOperator, ValueType

    bind_vars = {}

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
        doc_field = validate_field_name(doc_field, "arangodb")
    elif condition.right and condition.right.value_type == ValueType.DOCUMENT_FIELD:
        doc_field = ".".join(condition.right.field_path)
        doc_field = validate_field_name(doc_field, "arangodb")

    # Handle EXISTS
    if condition.operator == ConditionOperator.EXISTS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            return f"{doc_alias}.{doc_field} != null", {}

    # Handle NOT_EXISTS
    elif condition.operator == ConditionOperator.NOT_EXISTS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            return f"{doc_alias}.{doc_field} == null", {}

    # Handle EQUALS
    elif condition.operator == ConditionOperator.EQUALS:
        if (condition.left.value_type == ValueType.USER_FIELD and
                condition.right.value_type == ValueType.DOCUMENT_FIELD):
            if left_value is None:
                return "false", {}
            param_name = next_param("eq")
            bind_vars[param_name] = left_value
            return f"{doc_alias}.{doc_field} == @{param_name}", bind_vars

        elif condition.left.value_type == ValueType.DOCUMENT_FIELD:
            param_name = next_param("eq")
            bind_vars[param_name] = right_value
            return f"{doc_alias}.{doc_field} == @{param_name}", bind_vars

    # Handle NOT_EQUALS
    elif condition.operator == ConditionOperator.NOT_EQUALS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            param_name = next_param("neq")
            bind_vars[param_name] = right_value
            return f"{doc_alias}.{doc_field} != @{param_name}", bind_vars

    # Handle IN
    elif condition.operator == ConditionOperator.IN:
        if (condition.left.value_type == ValueType.USER_FIELD and
                condition.right.value_type == ValueType.DOCUMENT_FIELD):
            if left_value is None:
                return "false", {}
            param_name = next_param("in")
            bind_vars[param_name] = left_value
            # Check if user value is in document's array
            return f"@{param_name} IN {doc_alias}.{doc_field}", bind_vars

        elif (condition.left.value_type == ValueType.DOCUMENT_FIELD and
              right_value is not None and isinstance(right_value, list)):
            param_name = next_param("in")
            bind_vars[param_name] = right_value
            return f"{doc_alias}.{doc_field} IN @{param_name}", bind_vars

    # Handle NOT_IN
    elif condition.operator == ConditionOperator.NOT_IN:
        if (condition.left.value_type == ValueType.USER_FIELD and
                condition.right.value_type == ValueType.DOCUMENT_FIELD):
            if left_value is None:
                return "true", {}
            param_name = next_param("nin")
            bind_vars[param_name] = left_value
            return f"@{param_name} NOT IN {doc_alias}.{doc_field}", bind_vars

        elif (condition.left.value_type == ValueType.DOCUMENT_FIELD and
              right_value is not None and isinstance(right_value, list)):
            param_name = next_param("nin")
            bind_vars[param_name] = right_value
            return f"{doc_alias}.{doc_field} NOT IN @{param_name}", bind_vars

    # Handle comparison operators
    elif condition.operator == ConditionOperator.GREATER_THAN:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            param_name = next_param("gt")
            bind_vars[param_name] = right_value
            return f"{doc_alias}.{doc_field} > @{param_name}", bind_vars

    elif condition.operator == ConditionOperator.LESS_THAN:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            param_name = next_param("lt")
            bind_vars[param_name] = right_value
            return f"{doc_alias}.{doc_field} < @{param_name}", bind_vars

    elif condition.operator == ConditionOperator.GREATER_THAN_OR_EQUAL:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            param_name = next_param("gte")
            bind_vars[param_name] = right_value
            return f"{doc_alias}.{doc_field} >= @{param_name}", bind_vars

    elif condition.operator == ConditionOperator.LESS_THAN_OR_EQUAL:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            param_name = next_param("lte")
            bind_vars[param_name] = right_value
            return f"{doc_alias}.{doc_field} <= @{param_name}", bind_vars

    # If we reach here with an unhandled operator, raise an error instead of silently allowing
    raise UnsupportedConditionError(
        condition=str(condition.operator),
        backend="arangodb",
        reason=f"Operator {condition.operator} with value types "
               f"left={condition.left.value_type}, right={condition.right.value_type if condition.right else None} "
               f"is not supported. This is a security measure to prevent silent filter bypass."
    )


def _build_arangodb_condition_filter(
    condition: str,
    user: dict[str, Any],
    doc_alias: str,
    next_param: Callable[[str], str]
) -> Tuple[Optional[str], dict[str, Any]]:
    """
    Build an AQL condition from a condition expression.
    """
    from ...policy.compiler import CompiledExpression, ConditionCompiler

    try:
        compiled = ConditionCompiler.compile_expression(condition)

        if isinstance(compiled, CompiledExpression):
            return _build_arangodb_from_compiled_node(
                compiled, user, doc_alias, next_param
            )
        else:
            return _build_arangodb_from_condition(
                compiled, user, doc_alias, next_param
            )
    except (ValueError, AttributeError, KeyError, TypeError) as e:
        logger.debug(
            "Condition compilation failed, falling back to string parsing: %s (condition: %s)",
            str(e), condition
        )

    return _parse_condition_string(condition, user, doc_alias, next_param)


def _parse_condition_string(
    condition: str,
    user: dict[str, Any],
    doc_alias: str,
    next_param: Callable[[str], str]
) -> Tuple[Optional[str], dict[str, Any]]:
    """
    Parse a condition string and convert to AQL.
    """
    bind_vars = {}
    condition = condition.strip()

    # Parse field existence
    if " not exists" in condition:
        field = condition.replace(" not exists", "").strip()
        if field.startswith("document."):
            doc_field = validate_field_name(field[9:], "arangodb")
            return f"{doc_alias}.{doc_field} == null", {}
        return None, {}

    elif " exists" in condition:
        field = condition.replace(" exists", "").strip()
        if field.startswith("document."):
            doc_field = validate_field_name(field[9:], "arangodb")
            return f"{doc_alias}.{doc_field} != null", {}
        return None, {}

    # Parse !=
    elif "!=" in condition:
        parts = condition.split("!=", 1)
        if len(parts) != 2:
            return None, {}

        left, right = parts[0].strip(), parts[1].strip()

        if left.startswith("document."):
            doc_field = validate_field_name(left[9:], "arangodb")
            literal_value = parse_literal_value(right)
            param_name = next_param("neq")
            bind_vars[param_name] = literal_value
            return f"{doc_alias}.{doc_field} != @{param_name}", bind_vars

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
                return "false", {}

            if right.startswith("document."):
                doc_field = validate_field_name(right[9:], "arangodb")
                param_name = next_param("eq")
                bind_vars[param_name] = user_value
                return f"{doc_alias}.{doc_field} == @{param_name}", bind_vars

        elif left.startswith("document."):
            doc_field = validate_field_name(left[9:], "arangodb")
            literal_value = parse_literal_value(right)
            param_name = next_param("eq")
            bind_vars[param_name] = literal_value
            return f"{doc_alias}.{doc_field} == @{param_name}", bind_vars

    # Parse "not in"
    elif " not in " in condition:
        parts = condition.split(" not in ", 1)
        if len(parts) != 2:
            return None, {}

        left, right = parts[0].strip(), parts[1].strip()

        if left.startswith("user.") and right.startswith("document."):
            user_field = left[5:]
            doc_field = validate_field_name(right[9:], "arangodb")
            user_value = get_nested_value(user, user_field)

            if user_value is None:
                return "true", {}

            param_name = next_param("nin")
            bind_vars[param_name] = user_value
            return f"@{param_name} NOT IN {doc_alias}.{doc_field}", bind_vars

        elif left.startswith("document."):
            doc_field = validate_field_name(left[9:], "arangodb")
            list_values = parse_list_literal(right)

            if list_values is not None and len(list_values) > 0:
                param_name = next_param("nin")
                bind_vars[param_name] = list_values
                return f"{doc_alias}.{doc_field} NOT IN @{param_name}", bind_vars

    # Parse "in"
    elif " in " in condition:
        parts = condition.split(" in ", 1)
        if len(parts) != 2:
            return None, {}

        left, right = parts[0].strip(), parts[1].strip()

        if left.startswith("user.") and right.startswith("document."):
            user_field = left[5:]
            doc_field = validate_field_name(right[9:], "arangodb")
            user_value = get_nested_value(user, user_field)

            if user_value is None:
                return "false", {}

            param_name = next_param("in")
            bind_vars[param_name] = user_value
            return f"@{param_name} IN {doc_alias}.{doc_field}", bind_vars

        elif left.startswith("document."):
            doc_field = validate_field_name(left[9:], "arangodb")
            list_values = parse_list_literal(right)

            if list_values is not None:
                if len(list_values) == 0:
                    return "false", {}
                param_name = next_param("in")
                bind_vars[param_name] = list_values
                return f"{doc_alias}.{doc_field} IN @{param_name}", bind_vars

    return None, {}
