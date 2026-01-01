"""
Neo4j filter builder.

Converts RAGGuard policies to Cypher WHERE clauses for Neo4j.
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


def to_neo4j_filter(
    policy: Policy,
    user: dict[str, Any],
    node_alias: str = "doc"
) -> Tuple[str, dict[str, Any]]:
    """
    Convert a policy and user context to a Cypher WHERE clause.

    Args:
        policy: The access control policy
        user: User context
        node_alias: Alias for the document node in the query (default: "doc")

    Returns:
        Tuple of (where_clause, parameters)
        - where_clause: Cypher WHERE clause (without "WHERE" keyword)
        - parameters: Parameter dict for parameterized queries

    Example:
        >>> where_clause, params = to_neo4j_filter(policy, user)
        >>> query = f"MATCH (doc:Document) WHERE {where_clause} RETURN doc"
        >>> session.run(query, **params)
    """
    conditions = []
    params = {}
    param_counter = [0]  # Use list to allow mutation in nested function

    def next_param(prefix: str = "p") -> str:
        """Generate unique parameter name."""
        param_counter[0] += 1
        return f"{prefix}_{param_counter[0]}"

    for rule in policy.rules:
        rule_condition, rule_params = _build_neo4j_rule_filter(
            rule, user, node_alias, next_param
        )
        if rule_condition is not None:
            conditions.append(rule_condition)
            params.update(rule_params)

    if not conditions:
        # No rules apply to this user
        if policy.default == "allow":
            # Allow all - return empty filter (matches everything)
            return "", {}
        else:
            # Deny all - return filter that matches nothing
            # Use a condition that's always false
            return "false", {}

    if len(conditions) == 1:
        return conditions[0], params

    # Multiple rules: OR them together
    combined = " OR ".join(f"({c})" for c in conditions)
    return combined, params


def _build_neo4j_rule_filter(
    rule: Rule,
    user: dict[str, Any],
    node_alias: str,
    next_param: Callable[[str], str]
) -> Tuple[Optional[str], dict[str, Any]]:
    """
    Build a Cypher WHERE clause for a single rule.

    Returns:
        Tuple of (condition_string, parameters) or (None, {}) if rule doesn't apply
    """
    # Check if user satisfies the allow conditions
    if not user_satisfies_allow(rule.allow, user):
        return None, {}

    # User is allowed - build conditions
    conditions = []
    params = {}

    # Add match conditions
    if rule.match:
        for key, value in rule.match.items():
            # Validate field name
            safe_key = validate_field_name(key, "neo4j")

            if isinstance(value, list):
                # OR within the list
                param_name = next_param("match")
                params[param_name] = value
                conditions.append(f"{node_alias}.{safe_key} IN ${param_name}")
            else:
                param_name = next_param("match")
                params[param_name] = value
                conditions.append(f"{node_alias}.{safe_key} = ${param_name}")

    # Add conditions that reference user context
    if rule.allow.conditions:
        for condition in rule.allow.conditions:
            cond_str, cond_params = _build_neo4j_condition_filter(
                condition, user, node_alias, next_param
            )
            if cond_str:
                conditions.append(cond_str)
                params.update(cond_params)

    if not conditions:
        # No conditions - this rule allows access to all documents for this user
        return "true", {}

    if len(conditions) == 1:
        return conditions[0], params

    # Multiple conditions: AND them together
    combined = " AND ".join(conditions)
    return combined, params


def _build_neo4j_from_compiled_node(
    node: Union['CompiledCondition', 'CompiledExpression'],
    user: dict[str, Any],
    node_alias: str,
    next_param: Callable[[str], str]
) -> Tuple[Optional[str], dict[str, Any]]:
    """
    Build a Cypher condition from a compiled expression node.

    This enables native OR/AND support by converting the expression tree
    to Cypher's native logical operators.
    """
    from ...policy.compiler import CompiledCondition, CompiledExpression, LogicalOperator

    if isinstance(node, CompiledCondition):
        return _build_neo4j_from_condition(node, user, node_alias, next_param)

    elif isinstance(node, CompiledExpression):
        child_conditions = []
        params = {}

        for child in node.children:
            child_cond, child_params = _build_neo4j_from_compiled_node(
                child, user, node_alias, next_param
            )
            if child_cond is not None:
                child_conditions.append(child_cond)
                params.update(child_params)

        if not child_conditions:
            return None, {}

        if len(child_conditions) == 1:
            return child_conditions[0], params

        # Build native OR/AND condition
        if node.operator == LogicalOperator.OR:
            combined = " OR ".join(f"({c})" for c in child_conditions)
        else:  # AND
            combined = " AND ".join(child_conditions)

        return combined, params

    return None, {}


def _build_neo4j_from_condition(
    condition: 'CompiledCondition',
    user: dict[str, Any],
    node_alias: str,
    next_param: Callable[[str], str]
) -> Tuple[Optional[str], dict[str, Any]]:
    """
    Build a Cypher condition from a CompiledCondition.
    """
    from ...policy.compiler import CompiledConditionEvaluator, ConditionOperator, ValueType

    params = {}

    # Resolve user values at filter build time
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

    # Get document field path
    doc_field = None
    if condition.left.value_type == ValueType.DOCUMENT_FIELD:
        doc_field = ".".join(condition.left.field_path)
        doc_field = validate_field_name(doc_field, "neo4j")
    elif condition.right and condition.right.value_type == ValueType.DOCUMENT_FIELD:
        doc_field = ".".join(condition.right.field_path)
        doc_field = validate_field_name(doc_field, "neo4j")

    # Handle field existence operators
    if condition.operator == ConditionOperator.EXISTS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            return f"{node_alias}.{doc_field} IS NOT NULL", {}

    elif condition.operator == ConditionOperator.NOT_EXISTS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            return f"{node_alias}.{doc_field} IS NULL", {}

    # Handle EQUALS operator
    elif condition.operator == ConditionOperator.EQUALS:
        # user.field == document.field
        if (condition.left.value_type == ValueType.USER_FIELD and
                condition.right.value_type == ValueType.DOCUMENT_FIELD):
            if left_value is None:
                # User field is None, deny access
                return "false", {}
            param_name = next_param("eq")
            params[param_name] = left_value
            return f"{node_alias}.{doc_field} = ${param_name}", params

        # document.field == literal
        elif condition.left.value_type == ValueType.DOCUMENT_FIELD:
            param_name = next_param("eq")
            params[param_name] = right_value
            return f"{node_alias}.{doc_field} = ${param_name}", params

    # Handle NOT_EQUALS operator
    elif condition.operator == ConditionOperator.NOT_EQUALS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            param_name = next_param("neq")
            params[param_name] = right_value
            return f"{node_alias}.{doc_field} <> ${param_name}", params

    # Handle IN operator
    elif condition.operator == ConditionOperator.IN:
        # user.id in document.array_field
        if (condition.left.value_type == ValueType.USER_FIELD and
                condition.right.value_type == ValueType.DOCUMENT_FIELD):
            if left_value is None:
                return "false", {}
            param_name = next_param("in")
            params[param_name] = left_value
            return f"${param_name} IN {node_alias}.{doc_field}", params

        # document.field in [literals]
        elif (condition.left.value_type == ValueType.DOCUMENT_FIELD and
              right_value is not None and isinstance(right_value, list)):
            param_name = next_param("in")
            params[param_name] = right_value
            return f"{node_alias}.{doc_field} IN ${param_name}", params

    # Handle NOT_IN operator
    elif condition.operator == ConditionOperator.NOT_IN:
        # user.id not in document.array_field
        if (condition.left.value_type == ValueType.USER_FIELD and
                condition.right.value_type == ValueType.DOCUMENT_FIELD):
            if left_value is None:
                return "true", {}  # Not in anything if user field is None
            param_name = next_param("nin")
            params[param_name] = left_value
            return f"NOT ${param_name} IN {node_alias}.{doc_field}", params

        # document.field not in [literals]
        elif (condition.left.value_type == ValueType.DOCUMENT_FIELD and
              right_value is not None and isinstance(right_value, list)):
            param_name = next_param("nin")
            params[param_name] = right_value
            return f"NOT {node_alias}.{doc_field} IN ${param_name}", params

    # Handle comparison operators
    elif condition.operator in (
        ConditionOperator.GREATER_THAN,
        ConditionOperator.LESS_THAN,
        ConditionOperator.GREATER_THAN_OR_EQUAL,
        ConditionOperator.LESS_THAN_OR_EQUAL
    ):
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            op_map = {
                ConditionOperator.GREATER_THAN: ">",
                ConditionOperator.LESS_THAN: "<",
                ConditionOperator.GREATER_THAN_OR_EQUAL: ">=",
                ConditionOperator.LESS_THAN_OR_EQUAL: "<=",
            }
            op = op_map[condition.operator]
            param_name = next_param("cmp")
            params[param_name] = right_value
            return f"{node_alias}.{doc_field} {op} ${param_name}", params

    # If we reach here with an unhandled operator, raise an error instead of silently allowing
    raise UnsupportedConditionError(
        condition=str(condition.operator),
        backend="neo4j",
        reason=f"Operator {condition.operator} with value types "
               f"left={condition.left.value_type}, right={condition.right.value_type if condition.right else None} "
               f"is not supported. This is a security measure to prevent silent filter bypass."
    )


def _build_neo4j_condition_filter(
    condition: str,
    user: dict[str, Any],
    node_alias: str,
    next_param: Callable[[str], str]
) -> Tuple[Optional[str], dict[str, Any]]:
    """
    Build a Cypher condition from a condition expression.
    """
    from ...policy.compiler import CompiledExpression, ConditionCompiler

    # Try to compile as expression (supports OR/AND)
    try:
        compiled = ConditionCompiler.compile_expression(condition)

        if isinstance(compiled, CompiledExpression):
            return _build_neo4j_from_compiled_node(
                compiled, user, node_alias, next_param
            )
        else:
            return _build_neo4j_from_condition(
                compiled, user, node_alias, next_param
            )
    except (ValueError, AttributeError, KeyError, TypeError) as e:
        logger.debug(
            "Condition compilation failed, falling back to string parsing: %s (condition: %s)",
            str(e), condition
        )

    # Original string-based parsing (backward compatibility)
    return _parse_condition_string(condition, user, node_alias, next_param)


def _parse_condition_string(
    condition: str,
    user: dict[str, Any],
    node_alias: str,
    next_param: Callable[[str], str]
) -> Tuple[Optional[str], dict[str, Any]]:
    """
    Parse a condition string and convert to Cypher.

    Fallback for conditions that can't be compiled.
    """
    params = {}
    condition = condition.strip()

    # Parse field existence checks
    if " not exists" in condition:
        field = condition.replace(" not exists", "").strip()
        if field.startswith("document."):
            doc_field = validate_field_name(field[9:], "neo4j")
            return f"{node_alias}.{doc_field} IS NULL", {}
        return None, {}

    elif " exists" in condition:
        field = condition.replace(" exists", "").strip()
        if field.startswith("document."):
            doc_field = validate_field_name(field[9:], "neo4j")
            return f"{node_alias}.{doc_field} IS NOT NULL", {}
        return None, {}

    # Parse document.field != 'literal'
    elif "!=" in condition:
        parts = condition.split("!=", 1)
        if len(parts) != 2:
            return None, {}

        left, right = parts[0].strip(), parts[1].strip()

        if left.startswith("document."):
            doc_field = validate_field_name(left[9:], "neo4j")
            literal_value = parse_literal_value(right)
            param_name = next_param("neq")
            params[param_name] = literal_value
            return f"{node_alias}.{doc_field} <> ${param_name}", params

    # Parse user.field == document.field or document.field == 'literal'
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
                doc_field = validate_field_name(right[9:], "neo4j")
                param_name = next_param("eq")
                params[param_name] = user_value
                return f"{node_alias}.{doc_field} = ${param_name}", params

        elif left.startswith("document."):
            doc_field = validate_field_name(left[9:], "neo4j")
            literal_value = parse_literal_value(right)
            param_name = next_param("eq")
            params[param_name] = literal_value
            return f"{node_alias}.{doc_field} = ${param_name}", params

    # Parse not in conditions
    elif " not in " in condition:
        parts = condition.split(" not in ", 1)
        if len(parts) != 2:
            return None, {}

        left, right = parts[0].strip(), parts[1].strip()

        if left.startswith("user.") and right.startswith("document."):
            user_field = left[5:]
            doc_field = validate_field_name(right[9:], "neo4j")
            user_value = get_nested_value(user, user_field)

            if user_value is None:
                return "true", {}

            param_name = next_param("nin")
            params[param_name] = user_value
            return f"NOT ${param_name} IN {node_alias}.{doc_field}", params

        elif left.startswith("document."):
            doc_field = validate_field_name(left[9:], "neo4j")
            list_values = parse_list_literal(right)

            if list_values is not None and len(list_values) > 0:
                param_name = next_param("nin")
                params[param_name] = list_values
                return f"NOT {node_alias}.{doc_field} IN ${param_name}", params

    # Parse in conditions
    elif " in " in condition:
        parts = condition.split(" in ", 1)
        if len(parts) != 2:
            return None, {}

        left, right = parts[0].strip(), parts[1].strip()

        if left.startswith("user.") and right.startswith("document."):
            user_field = left[5:]
            doc_field = validate_field_name(right[9:], "neo4j")
            user_value = get_nested_value(user, user_field)

            if user_value is None:
                return "false", {}

            param_name = next_param("in")
            params[param_name] = user_value
            return f"${param_name} IN {node_alias}.{doc_field}", params

        elif left.startswith("document."):
            doc_field = validate_field_name(left[9:], "neo4j")
            list_values = parse_list_literal(right)

            if list_values is not None:
                if len(list_values) == 0:
                    return "false", {}
                param_name = next_param("in")
                params[param_name] = list_values
                return f"{node_alias}.{doc_field} IN ${param_name}", params

    return None, {}
