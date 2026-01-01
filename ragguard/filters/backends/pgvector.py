"""
pgvector (PostgreSQL) filter builder.

Converts RAGGuard policies to SQL WHERE clauses for pgvector.
"""

from typing import TYPE_CHECKING, Any, Union

from ...exceptions import UnsupportedConditionError
from ...policy.models import Policy, Rule
from ..base import (
    get_nested_value,
    logger,
    parse_list_literal,
    parse_literal_value,
    user_satisfies_allow,
    validate_field_name,
    validate_field_path,
    validate_sql_identifier,
)

if TYPE_CHECKING:
    from ...policy.compiler import CompiledCondition, CompiledExpression


def _validate_policy_fields_upfront(policy: Policy) -> None:
    """
    Validate all field names in the policy upfront before SQL generation.

    This prevents partial SQL generation before a validation failure,
    ensuring all-or-nothing validation for security.

    Args:
        policy: The access control policy

    Raises:
        ValueError: If any field name is invalid
    """
    for rule in policy.rules:
        # Validate match fields
        if rule.match:
            for key in rule.match:
                validate_sql_identifier(key, f"match field '{key}'")

        # Validate fields in conditions (extract field names from condition strings)
        if rule.allow and rule.allow.conditions:
            for condition in rule.allow.conditions:
                _validate_condition_fields(condition)


def _validate_condition_fields(condition: str) -> None:
    """
    Validate field names in a condition string.

    Args:
        condition: The condition string to validate

    Raises:
        ValueError: If any field name is invalid
    """
    import re

    # Extract document.field references
    doc_fields = re.findall(r'document\.([a-zA-Z_][a-zA-Z0-9_]*)', condition)
    for field in doc_fields:
        validate_field_name(field, "pgvector")


def to_pgvector_filter(
    policy: Policy,
    user: dict[str, Any],
    metadata_column: str = "metadata"
) -> tuple[str, list[Any]]:
    """
    Convert a policy and user context to a pgvector WHERE clause.

    Returns a tuple of (where_clause, params) for parameterized queries
    to prevent SQL injection.

    Args:
        policy: The access control policy
        user: User context
        metadata_column: Name of the JSONB column containing metadata (default: "metadata")

    Returns:
        Tuple of (WHERE clause string, list of parameters)

    Raises:
        ValueError: If any field name in the policy is invalid (SQL injection prevention)
    """
    # Validate SQL identifier to prevent injection
    validate_sql_identifier(metadata_column, "metadata_column")

    # SECURITY: Validate ALL field names upfront before any SQL generation
    # This prevents partial SQL construction if validation fails mid-generation
    _validate_policy_fields_upfront(policy)

    where_clauses = []  # OR between rules
    params = []

    for rule in policy.rules:
        rule_clause, rule_params = _build_pgvector_rule_filter(rule, user, metadata_column)
        if rule_clause:
            where_clauses.append(rule_clause)
            params.extend(rule_params)

    if not where_clauses:
        # No rules apply to this user
        if policy.default == "allow":
            return ("", [])
        else:
            return ("WHERE FALSE", [])

    # Combine with OR
    combined = " OR ".join(f"({clause})" for clause in where_clauses)
    return (f"WHERE {combined}", params)


def _build_pgvector_rule_filter(
    rule: Rule,
    user: dict[str, Any],
    metadata_column: str = "metadata"
) -> tuple[str, list[Any]]:
    """Build a pgvector WHERE clause for a single rule."""
    if not user_satisfies_allow(rule.allow, user):
        return ("", [])

    conditions = []
    params = []

    # Add match conditions
    if rule.match:
        for key, value in rule.match.items():
            validate_sql_identifier(key, f"match field '{key}'")

            if isinstance(value, list):
                placeholders = ", ".join(["%s"] * len(value))
                conditions.append(f"{key} IN ({placeholders})")
                params.extend(value)
            else:
                conditions.append(f"{key} = %s")
                params.append(value)

    # Add conditions that reference user context
    if rule.allow.conditions:
        for condition in rule.allow.conditions:
            cond_clause, cond_params = _build_pgvector_condition_filter(
                condition, user, metadata_column
            )
            if cond_clause:
                conditions.append(cond_clause)
                params.extend(cond_params)

    if not conditions:
        return ("TRUE", [])

    combined = " AND ".join(conditions)
    return (combined, params)


def _build_pgvector_from_compiled_node(
    node: Union['CompiledCondition', 'CompiledExpression'],
    user: dict[str, Any],
    metadata_column: str = "metadata"
) -> tuple[str, list[Any]]:
    """
    Build pgvector SQL from a compiled expression node.

    Enables native OR/AND support by converting expression trees to SQL.
    """
    from ...policy.compiler import CompiledCondition, CompiledExpression, LogicalOperator

    if isinstance(node, CompiledCondition):
        return _build_pgvector_from_condition(node, user, metadata_column)

    elif isinstance(node, CompiledExpression):
        child_clauses = []
        child_params = []

        for child in node.children:
            sql, params = _build_pgvector_from_compiled_node(child, user, metadata_column)
            if sql:
                child_clauses.append(sql)
                child_params.extend(params)

        if not child_clauses:
            return ("", [])

        if len(child_clauses) == 1:
            return (child_clauses[0], child_params)

        if node.operator == LogicalOperator.OR:
            combined = " OR ".join(f"({clause})" for clause in child_clauses)
            return (combined, child_params)
        elif node.operator == LogicalOperator.AND:
            combined = " AND ".join(f"({clause})" for clause in child_clauses)
            return (combined, child_params)

    return ("", [])


def _build_pgvector_from_condition(
    condition: 'CompiledCondition',
    user: dict[str, Any],
    metadata_column: str = "metadata"
) -> tuple[str, list[Any]]:
    """
    Build pgvector SQL from a CompiledCondition.

    Converts the compiled AST to SQL WHERE clause.
    """
    from ...policy.compiler import CompiledConditionEvaluator, ConditionOperator, ValueType

    def _pgvector_field(field_path: list[str], as_text: bool = True) -> str:
        """Convert field path to pgvector JSONB accessor."""
        # Validate field path to prevent SQL injection
        field_name = validate_field_path(field_path, "pgvector")
        if as_text:
            return f"{metadata_column}->>'{field_name}'"
        else:
            return f"{metadata_column}->'{field_name}'"

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

    # Handle field existence operators
    if condition.operator == ConditionOperator.EXISTS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = _pgvector_field(condition.left.field_path)
            return (f"{doc_field} IS NOT NULL", [])

    elif condition.operator == ConditionOperator.NOT_EXISTS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = _pgvector_field(condition.left.field_path)
            return (f"{doc_field} IS NULL", [])

    # Handle EQUALS operator
    elif condition.operator == ConditionOperator.EQUALS:
        if condition.left.value_type == ValueType.USER_FIELD and condition.right.value_type == ValueType.DOCUMENT_FIELD:
            if left_value is None:
                return ("FALSE", [])
            doc_field = _pgvector_field(condition.right.field_path, as_text=True)
            if isinstance(left_value, bool):
                return (f"({doc_field})::boolean = %s", [left_value])
            elif isinstance(left_value, (int, float)):
                return (f"({doc_field})::numeric = %s", [left_value])
            else:
                return (f"{doc_field} = %s", [left_value])

        # user.field == literal (e.g., user.role == 'admin')
        # This is evaluated at filter-build time, not as a database filter
        elif condition.left.value_type == ValueType.USER_FIELD and condition.right.value_type in (ValueType.LITERAL_STRING, ValueType.LITERAL_NUMBER, ValueType.LITERAL_BOOL):
            if left_value == right_value:
                return ("TRUE", [])  # Condition is true - no filter needed
            else:
                return ("FALSE", [])  # Condition is false - deny all

        elif condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = _pgvector_field(condition.left.field_path, as_text=True)
            if isinstance(right_value, bool):
                return (f"({doc_field})::boolean = %s", [right_value])
            elif isinstance(right_value, (int, float)):
                return (f"({doc_field})::numeric = %s", [right_value])
            else:
                return (f"{doc_field} = %s", [right_value])

    # Handle NOT_EQUALS operator
    elif condition.operator == ConditionOperator.NOT_EQUALS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = _pgvector_field(condition.left.field_path, as_text=True)
            if isinstance(right_value, bool):
                return (f"({doc_field})::boolean != %s", [right_value])
            elif isinstance(right_value, (int, float)):
                return (f"({doc_field})::numeric != %s", [right_value])
            else:
                return (f"{doc_field} != %s", [right_value])

    # Handle IN operator
    elif condition.operator == ConditionOperator.IN:
        if condition.left.value_type == ValueType.USER_FIELD and condition.right.value_type == ValueType.DOCUMENT_FIELD:
            if left_value is None:
                return ("FALSE", [])
            doc_field_name = validate_field_path(condition.right.field_path, "pgvector")
            return (f"%s::text = ANY(SELECT jsonb_array_elements_text({metadata_column}->'{doc_field_name}'))", [left_value])

        elif condition.left.value_type == ValueType.DOCUMENT_FIELD and right_value is not None and isinstance(right_value, list):
            doc_field = _pgvector_field(condition.left.field_path)
            if len(right_value) == 0:
                return ("FALSE", [])
            placeholders = ", ".join(["%s"] * len(right_value))
            return (f"{doc_field} IN ({placeholders})", right_value)

        # Handle literal IN document.array (e.g., 'public' in document.tags)
        elif condition.left.value_type == ValueType.LITERAL_STRING and condition.right.value_type == ValueType.DOCUMENT_FIELD:
            literal_value = condition.left.value
            doc_field_name = validate_field_path(condition.right.field_path, "pgvector")
            return (f"%s = ANY(SELECT jsonb_array_elements_text({metadata_column}->'{doc_field_name}'))", [literal_value])

    # Handle NOT_IN operator
    elif condition.operator == ConditionOperator.NOT_IN:
        if condition.left.value_type == ValueType.USER_FIELD and condition.right.value_type == ValueType.DOCUMENT_FIELD:
            if left_value is None:
                return ("TRUE", [])
            doc_field_name = validate_field_path(condition.right.field_path, "pgvector")
            return (f"NOT (%s::text = ANY(SELECT jsonb_array_elements_text({metadata_column}->'{doc_field_name}')))", [left_value])

        elif condition.left.value_type == ValueType.DOCUMENT_FIELD and right_value is not None and isinstance(right_value, list):
            doc_field = _pgvector_field(condition.left.field_path)
            if len(right_value) == 0:
                return ("TRUE", [])
            placeholders = ", ".join(["%s"] * len(right_value))
            return (f"{doc_field} NOT IN ({placeholders})", right_value)

        # Handle literal NOT IN document.array (e.g., 'archived' not in document.tags)
        elif condition.left.value_type == ValueType.LITERAL_STRING and condition.right.value_type == ValueType.DOCUMENT_FIELD:
            literal_value = condition.left.value
            doc_field_name = validate_field_path(condition.right.field_path, "pgvector")
            return (f"NOT (%s = ANY(SELECT jsonb_array_elements_text({metadata_column}->'{doc_field_name}')))", [literal_value])

    # Handle comparison operators
    elif condition.operator == ConditionOperator.GREATER_THAN:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = _pgvector_field(condition.left.field_path)
            return (f"({doc_field})::numeric > %s", [right_value])

    elif condition.operator == ConditionOperator.LESS_THAN:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = _pgvector_field(condition.left.field_path)
            return (f"({doc_field})::numeric < %s", [right_value])

    elif condition.operator == ConditionOperator.GREATER_THAN_OR_EQUAL:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = _pgvector_field(condition.left.field_path)
            return (f"({doc_field})::numeric >= %s", [right_value])

    elif condition.operator == ConditionOperator.LESS_THAN_OR_EQUAL:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = _pgvector_field(condition.left.field_path)
            return (f"({doc_field})::numeric <= %s", [right_value])

    # Handle comparison operators for user.field vs literal (e.g., user.level >= 5)
    # Check if this is a user field vs literal comparison for any comparison operator
    if condition.operator in (ConditionOperator.GREATER_THAN, ConditionOperator.LESS_THAN,
                              ConditionOperator.GREATER_THAN_OR_EQUAL, ConditionOperator.LESS_THAN_OR_EQUAL):
        if condition.left.value_type == ValueType.USER_FIELD and condition.right.value_type == ValueType.LITERAL_NUMBER:
            comparison_result = False
            if left_value is not None and right_value is not None:
                if condition.operator == ConditionOperator.GREATER_THAN:
                    comparison_result = left_value > right_value
                elif condition.operator == ConditionOperator.GREATER_THAN_OR_EQUAL:
                    comparison_result = left_value >= right_value
                elif condition.operator == ConditionOperator.LESS_THAN:
                    comparison_result = left_value < right_value
                elif condition.operator == ConditionOperator.LESS_THAN_OR_EQUAL:
                    comparison_result = left_value <= right_value

            if comparison_result:
                return ("TRUE", [])  # Condition is true - no filter needed
            else:
                return ("FALSE", [])  # Condition is false - deny all

    # If we reach here with an unhandled operator, raise an error instead of silently allowing
    raise UnsupportedConditionError(
        condition=str(condition.operator),
        backend="pgvector",
        reason=f"Operator {condition.operator} with value types "
               f"left={condition.left.value_type}, right={condition.right.value_type if condition.right else None} "
               f"is not supported. This is a security measure to prevent silent filter bypass."
    )


def _build_pgvector_condition_filter(
    condition: str,
    user: dict[str, Any],
    metadata_column: str = "metadata"
) -> tuple[str, list[Any]]:
    """
    Build a SQL condition from a policy condition expression.

    v0.3.0: Now supports native OR/AND logic by detecting CompiledExpression.
    """
    from ...policy.compiler import CompiledCondition, CompiledExpression, ConditionCompiler

    # v0.3.0: Try to compile as expression (supports OR/AND)
    try:
        compiled = ConditionCompiler.compile_expression(condition)

        if isinstance(compiled, (CompiledExpression, CompiledCondition)):
            return _build_pgvector_from_compiled_node(compiled, user, metadata_column)
    except (ValueError, AttributeError, KeyError, TypeError):
        pass

    # Original string-based parsing (backward compatibility)
    condition = condition.strip()

    # Wrap in try/except to handle malformed conditions securely
    # SECURITY: Fail-closed - invalid conditions result in deny-all rather than silent bypass
    try:
        return _parse_pgvector_legacy_condition(condition, user, metadata_column)
    except ValueError as e:
        # Invalid field name or other validation error - fail-closed with deny-all
        logger.warning(f"Invalid condition '{condition}' - applying deny-all filter: {e}")
        raise UnsupportedConditionError(
            condition=condition,
            backend="pgvector",
            reason=f"{e}. This is a security measure to prevent filter bypass."
        )


def _parse_pgvector_legacy_condition(
    condition: str,
    user: dict[str, Any],
    metadata_column: str = "metadata"
) -> tuple[str, list[Any]]:
    """Parse legacy string condition format (internal helper)."""
    # Parse field existence checks
    if " not exists" in condition:
        field = condition.replace(" not exists", "").strip()
        if field.startswith("document."):
            doc_field = field[9:]
            validate_field_name(doc_field, "pgvector")
            return (f"{doc_field} IS NULL", [])
        return ("", [])

    elif " exists" in condition:
        field = condition.replace(" exists", "").strip()
        if field.startswith("document."):
            doc_field = field[9:]
            validate_field_name(doc_field, "pgvector")
            return (f"{doc_field} IS NOT NULL", [])
        return ("", [])

    # Parse document.field != 'literal' (negation)
    elif "!=" in condition:
        parts = condition.split("!=", 1)
        if len(parts) != 2:
            return ("", [])

        left, right = parts[0].strip(), parts[1].strip()

        if left.startswith("document."):
            doc_field = left[9:]
            validate_field_name(doc_field, "pgvector")
            literal_value = parse_literal_value(right)
            return (f"{doc_field} != %s", [literal_value])

    # Parse user.field == document.field or document.field == 'literal'
    elif "==" in condition:
        parts = condition.split("==", 1)
        if len(parts) != 2:
            return ("", [])

        left, right = parts[0].strip(), parts[1].strip()

        if left.startswith("user.") and right.startswith("document."):
            user_field = left[5:]
            doc_field = right[9:]
            validate_field_name(doc_field, "pgvector")
            user_value = get_nested_value(user, user_field)
            return (f"{doc_field} = %s", [user_value])

        elif left.startswith("document."):
            doc_field = left[9:]
            validate_field_name(doc_field, "pgvector")
            literal_value = parse_literal_value(right)
            return (f"{doc_field} = %s", [literal_value])

    # Parse not in conditions
    elif " not in " in condition:
        parts = condition.split(" not in ", 1)
        if len(parts) != 2:
            return ("", [])

        left, right = parts[0].strip(), parts[1].strip()

        if left.startswith("user.") and right.startswith("document."):
            user_field = left[5:]
            doc_field = right[9:]
            validate_field_name(doc_field, "pgvector")
            user_value = get_nested_value(user, user_field)

            if user_value is None:
                return ("1 = 1", [])

            return (f"NOT (%s = ANY({doc_field}))", [user_value])

        elif right.startswith("document."):
            doc_field = right[9:]
            validate_field_name(doc_field, "pgvector")
            literal_value = parse_literal_value(left)
            return (f"NOT (%s = ANY({doc_field}))", [literal_value])

        elif left.startswith("document."):
            doc_field = left[9:]
            validate_field_name(doc_field, "pgvector")
            list_values = parse_list_literal(right)

            if list_values is not None:
                if len(list_values) == 0:
                    return ("1 = 1", [])
                else:
                    placeholders = ", ".join(["%s"] * len(list_values))
                    return (f"{doc_field} NOT IN ({placeholders})", list_values)

    # Parse in conditions
    elif " in " in condition:
        parts = condition.split(" in ")
        if len(parts) != 2:
            return ("", [])

        left, right = parts[0].strip(), parts[1].strip()

        if left.startswith("user.") and right.startswith("document."):
            user_field = left[5:]
            doc_field = right[9:]
            validate_field_name(doc_field, "pgvector")
            user_value = get_nested_value(user, user_field)
            return (f"%s = ANY({doc_field})", [user_value])

        elif right.startswith("document."):
            doc_field = right[9:]
            validate_field_name(doc_field, "pgvector")
            literal_value = parse_literal_value(left)
            return (f"%s = ANY({doc_field})", [literal_value])

        elif left.startswith("document."):
            doc_field = left[9:]
            validate_field_name(doc_field, "pgvector")
            list_values = parse_list_literal(right)

            if list_values is not None:
                if len(list_values) == 0:
                    return ("1 = 0", [])
                else:
                    placeholders = ", ".join(["%s"] * len(list_values))
                    return (f"{doc_field} IN ({placeholders})", list_values)

    return ("", [])
