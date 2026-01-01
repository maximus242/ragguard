"""
Qdrant filter builder.

Converts RAGGuard policies to Qdrant-native filter format.
"""

from typing import TYPE_CHECKING, Any, Optional, Union

from ...exceptions import FilterBuildError, UnsupportedConditionError
from ...policy.models import Policy, Rule
from ..base import (
    DENY_ALL_FIELD,
    DENY_ALL_VALUE,
    get_nested_value,
    logger,
    parse_list_literal,
    parse_literal_value,
    user_satisfies_allow,
)

if TYPE_CHECKING:
    from ...policy.compiler import CompiledCondition, CompiledExpression


def to_qdrant_filter(policy: Policy, user: dict[str, Any]) -> Any:
    """
    Convert a policy and user context to a Qdrant Filter.

    The filter uses OR logic between rules (any matching rule grants access)
    and AND logic within a rule (all conditions must be satisfied).

    Args:
        policy: The access control policy
        user: User context

    Returns:
        qdrant_client.models.Filter object
    """
    try:
        from qdrant_client import models
    except ImportError:
        raise FilterBuildError(
            "qdrant-client not installed. Install with: pip install ragguard[qdrant]"
        )

    should_filters = []  # OR between rules

    for rule in policy.rules:
        rule_filter = _build_qdrant_rule_filter(rule, user, models)
        if rule_filter is not None:
            should_filters.append(rule_filter)

    if not should_filters:
        # No rules apply to this user
        if policy.default == "allow":
            # Allow all - return empty filter
            return None
        else:
            # Deny all - return filter that matches nothing
            return models.Filter(
                must=[
                    models.FieldCondition(
                        key=DENY_ALL_FIELD,
                        match=models.MatchValue(value=DENY_ALL_VALUE)
                    )
                ]
            )

    if len(should_filters) == 1:
        return should_filters[0]

    # Multiple rules: OR them together
    return models.Filter(should=should_filters)


def _build_qdrant_rule_filter(
    rule: Rule,
    user: dict[str, Any],
    models: Any
) -> Optional[Any]:
    """
    Build a Qdrant filter for a single rule.

    Converts a RAGGuard rule into a Qdrant Filter object. Returns None
    if the user doesn't satisfy the rule's allow conditions.
    """
    # Check if user satisfies the allow conditions
    if not user_satisfies_allow(rule.allow, user):
        return None

    # User is allowed - build filter for document match conditions
    must_conditions = []

    # Add match conditions (what documents this rule applies to)
    if rule.match:
        for key, value in rule.match.items():
            if isinstance(value, list):
                # OR within the list
                should_values = [
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=v)
                    )
                    for v in value
                ]
                must_conditions.append(models.Filter(should=should_values))
            else:
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )

    # Add conditions that reference user context
    if rule.allow.conditions:
        for condition in rule.allow.conditions:
            condition_filter = _build_qdrant_condition_filter(
                condition, user, models
            )
            if condition_filter:
                must_conditions.append(condition_filter)

    if not must_conditions:
        # No match conditions - this rule applies to all documents for this user
        return models.Filter(must=[])

    if len(must_conditions) == 1:
        # Single condition - return it directly or wrap if it's a FieldCondition
        if isinstance(must_conditions[0], models.FieldCondition):
            return models.Filter(must=[must_conditions[0]])
        return must_conditions[0]

    # Multiple conditions: AND them together
    return models.Filter(must=must_conditions)


def _build_qdrant_from_compiled_node(
    node: Union['CompiledCondition', 'CompiledExpression'],
    user: dict[str, Any],
    models: Any
) -> Optional[Any]:
    """
    Build a Qdrant filter from a compiled expression node.

    This enables native OR/AND support by converting the expression tree
    to Qdrant's native filter format.
    """
    from ...policy.compiler import CompiledCondition, CompiledExpression, LogicalOperator

    if isinstance(node, CompiledCondition):
        return _build_qdrant_from_condition(node, user, models)

    elif isinstance(node, CompiledExpression):
        child_filters = []

        for child in node.children:
            child_filter = _build_qdrant_from_compiled_node(child, user, models)
            if child_filter is not None:
                child_filters.append(child_filter)

        if not child_filters:
            return None

        if len(child_filters) == 1:
            return child_filters[0]

        # Build native OR/AND filter
        if node.operator == LogicalOperator.OR:
            return models.Filter(should=child_filters)
        elif node.operator == LogicalOperator.AND:
            return models.Filter(must=child_filters)

    return None


def _build_qdrant_from_condition(
    condition: 'CompiledCondition',
    user: dict[str, Any],
    models: Any
) -> Optional[Any]:
    """
    Build a Qdrant filter from a CompiledCondition.

    Converts the compiled AST node to native Qdrant filter format.
    """
    from ...policy.compiler import CompiledConditionEvaluator, ConditionOperator, ValueType

    # Resolve user values at filter build time
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
            doc_field = ".".join(condition.left.field_path)
            return models.Filter(
                must_not=[
                    models.IsEmptyCondition(
                        is_empty=models.PayloadField(key=doc_field)
                    )
                ]
            )

    elif condition.operator == ConditionOperator.NOT_EXISTS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = ".".join(condition.left.field_path)
            return models.Filter(
                must=[
                    models.IsEmptyCondition(
                        is_empty=models.PayloadField(key=doc_field)
                    )
                ]
            )

    # Handle EQUALS operator
    elif condition.operator == ConditionOperator.EQUALS:
        # user.field == document.field
        if condition.left.value_type == ValueType.USER_FIELD and condition.right.value_type == ValueType.DOCUMENT_FIELD:
            if left_value is None:
                return models.FieldCondition(
                    key=DENY_ALL_FIELD,
                    match=models.MatchValue(value=DENY_ALL_VALUE)
                )
            doc_field = ".".join(condition.right.field_path)
            return models.FieldCondition(
                key=doc_field,
                match=models.MatchValue(value=left_value)
            )

        # user.field == literal (e.g., user.role == 'admin')
        # This is evaluated at filter-build time, not as a database filter
        elif condition.left.value_type == ValueType.USER_FIELD and condition.right.value_type in (ValueType.LITERAL_STRING, ValueType.LITERAL_NUMBER, ValueType.LITERAL_BOOL):
            if left_value == right_value:
                # Condition is true - no filter needed (allow)
                return None
            else:
                # Condition is false - deny all
                return models.FieldCondition(
                    key=DENY_ALL_FIELD,
                    match=models.MatchValue(value=DENY_ALL_VALUE)
                )

        # document.field == literal
        elif condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = ".".join(condition.left.field_path)
            return models.FieldCondition(
                key=doc_field,
                match=models.MatchValue(value=right_value)
            )

    # Handle NOT_EQUALS operator
    elif condition.operator == ConditionOperator.NOT_EQUALS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = ".".join(condition.left.field_path)
            return models.Filter(
                must_not=[
                    models.FieldCondition(
                        key=doc_field,
                        match=models.MatchValue(value=right_value)
                    )
                ]
            )

    # Handle IN operator
    elif condition.operator == ConditionOperator.IN:
        # user.id in document.array_field
        if condition.left.value_type == ValueType.USER_FIELD and condition.right.value_type == ValueType.DOCUMENT_FIELD:
            if left_value is None:
                return models.FieldCondition(
                    key=DENY_ALL_FIELD,
                    match=models.MatchValue(value=DENY_ALL_VALUE)
                )
            doc_field = ".".join(condition.right.field_path)
            return models.FieldCondition(
                key=doc_field,
                match=models.MatchAny(any=[left_value])
            )

        # document.field in [literals]
        elif condition.left.value_type == ValueType.DOCUMENT_FIELD and right_value is not None and isinstance(right_value, list):
            doc_field = ".".join(condition.left.field_path)
            should_filters = [
                models.FieldCondition(
                    key=doc_field,
                    match=models.MatchValue(value=v)
                )
                for v in right_value
            ]
            return models.Filter(should=should_filters)

    # Handle NOT_IN operator
    elif condition.operator == ConditionOperator.NOT_IN:
        # user.id not in document.array_field
        if condition.left.value_type == ValueType.USER_FIELD and condition.right.value_type == ValueType.DOCUMENT_FIELD:
            if left_value is None:
                return None
            doc_field = ".".join(condition.right.field_path)
            return models.Filter(
                must_not=[
                    models.FieldCondition(
                        key=doc_field,
                        match=models.MatchAny(any=[left_value])
                    )
                ]
            )

        # document.field not in [literals]
        elif condition.left.value_type == ValueType.DOCUMENT_FIELD and right_value is not None and isinstance(right_value, list):
            doc_field = ".".join(condition.left.field_path)
            must_not_filters = [
                models.FieldCondition(
                    key=doc_field,
                    match=models.MatchValue(value=v)
                )
                for v in right_value
            ]
            return models.Filter(must_not=must_not_filters)

    # Handle comparison operators
    elif condition.operator in (ConditionOperator.GREATER_THAN, ConditionOperator.LESS_THAN,
                                  ConditionOperator.GREATER_THAN_OR_EQUAL, ConditionOperator.LESS_THAN_OR_EQUAL):
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = ".".join(condition.left.field_path)

            range_params = {}
            if condition.operator == ConditionOperator.GREATER_THAN:
                range_params["gt"] = right_value
            elif condition.operator == ConditionOperator.GREATER_THAN_OR_EQUAL:
                range_params["gte"] = right_value
            elif condition.operator == ConditionOperator.LESS_THAN:
                range_params["lt"] = right_value
            elif condition.operator == ConditionOperator.LESS_THAN_OR_EQUAL:
                range_params["lte"] = right_value

            return models.FieldCondition(
                key=doc_field,
                range=models.Range(**range_params)
            )

        # user.field >= literal (e.g., user.level >= 5)
        # This is evaluated at filter-build time, not as a database filter
        elif condition.left.value_type == ValueType.USER_FIELD and condition.right.value_type == ValueType.LITERAL_NUMBER:
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
                return None  # Condition is true - no filter needed
            else:
                return models.FieldCondition(
                    key=DENY_ALL_FIELD,
                    match=models.MatchValue(value=DENY_ALL_VALUE)
                )

    # If we reach here with an unhandled operator, raise an error instead of silently allowing
    # This prevents security bypass from unsupported operators
    raise UnsupportedConditionError(
        condition=str(condition.operator),
        backend="qdrant",
        reason=f"Operator {condition.operator} with value types "
               f"left={condition.left.value_type}, right={condition.right.value_type if condition.right else None} "
               f"is not supported. This is a security measure to prevent silent filter bypass."
    )


def _build_qdrant_condition_filter(
    condition: str,
    user: dict[str, Any],
    models: Any
) -> Optional[Any]:
    """
    Build a Qdrant filter from a condition expression.

    v0.3.0: Now supports native OR/AND logic by detecting CompiledExpression
    and routing to native filter builder.
    """
    from ...policy.compiler import CompiledExpression, ConditionCompiler

    # v0.3.0: Try to compile as expression (supports OR/AND)
    try:
        compiled = ConditionCompiler.compile_expression(condition)

        if isinstance(compiled, CompiledExpression):
            return _build_qdrant_from_compiled_node(compiled, user, models)
    except (ValueError, AttributeError, KeyError, TypeError) as e:
        logger.debug(
            "Condition compilation failed, falling back to string parsing: %s (condition: %s)",
            str(e), condition
        )

    # Original string-based parsing (backward compatibility)
    condition = condition.strip()

    # Parse field existence checks
    if " not exists" in condition:
        field = condition.replace(" not exists", "").strip()
        if field.startswith("document."):
            doc_field = field[9:]
            return models.Filter(
                must=[
                    models.IsEmptyCondition(
                        is_empty=models.PayloadField(key=doc_field)
                    )
                ]
            )
        return None

    elif " exists" in condition:
        field = condition.replace(" exists", "").strip()
        if field.startswith("document."):
            doc_field = field[9:]
            return models.Filter(
                must_not=[
                    models.IsEmptyCondition(
                        is_empty=models.PayloadField(key=doc_field)
                    )
                ]
            )
        return None

    # Parse document.field != 'literal' (negation)
    elif "!=" in condition:
        parts = condition.split("!=", 1)
        if len(parts) != 2:
            return None

        left, right = parts[0].strip(), parts[1].strip()

        if left.startswith("document."):
            doc_field = left[9:]
            literal_value = parse_literal_value(right)

            return models.Filter(
                must_not=[
                    models.FieldCondition(
                        key=doc_field,
                        match=models.MatchValue(value=literal_value)
                    )
                ]
            )

    # Parse user.field == document.field or document.field == 'literal'
    elif "==" in condition:
        parts = condition.split("==", 1)
        if len(parts) != 2:
            return None

        left, right = parts[0].strip(), parts[1].strip()

        if left.startswith("user."):
            user_field = left[5:]
            user_value = get_nested_value(user, user_field)

            if user_value is None:
                return models.FieldCondition(
                    key=DENY_ALL_FIELD,
                    match=models.MatchValue(value=DENY_ALL_VALUE)
                )

            if right.startswith("document."):
                doc_field = right[9:]
                return models.FieldCondition(
                    key=doc_field,
                    match=models.MatchValue(value=user_value)
                )

        elif left.startswith("document."):
            doc_field = left[9:]
            literal_value = parse_literal_value(right)

            return models.FieldCondition(
                key=doc_field,
                match=models.MatchValue(value=literal_value)
            )

    # Parse not in conditions
    elif " not in " in condition:
        parts = condition.split(" not in ", 1)
        if len(parts) != 2:
            return None

        left, right = parts[0].strip(), parts[1].strip()

        if left.startswith("user.") and right.startswith("document."):
            user_field = left[5:]
            doc_field = right[9:]
            user_value = get_nested_value(user, user_field)

            if user_value is None:
                return None

            return models.Filter(
                must_not=[
                    models.FieldCondition(
                        key=doc_field,
                        match=models.MatchAny(any=[user_value])
                    )
                ]
            )

        elif right.startswith("document."):
            doc_field = right[9:]
            literal_value = parse_literal_value(left)

            return models.Filter(
                must_not=[
                    models.FieldCondition(
                        key=doc_field,
                        match=models.MatchAny(any=[literal_value])
                    )
                ]
            )

        elif left.startswith("document."):
            doc_field = left[9:]
            list_values = parse_list_literal(right)

            if list_values is not None:
                if len(list_values) == 0:
                    return None
                else:
                    return models.Filter(
                        must_not=[
                            models.FieldCondition(
                                key=doc_field,
                                match=models.MatchAny(any=list_values)
                            )
                        ]
                    )

    # Parse in conditions
    elif " in " in condition:
        parts = condition.split(" in ")
        if len(parts) != 2:
            return None

        left, right = parts[0].strip(), parts[1].strip()

        if left.startswith("user.") and right.startswith("document."):
            user_field = left[5:]
            doc_field = right[9:]
            user_value = get_nested_value(user, user_field)

            if user_value is None:
                return models.FieldCondition(
                    key=DENY_ALL_FIELD,
                    match=models.MatchValue(value=DENY_ALL_VALUE)
                )

            return models.FieldCondition(
                key=doc_field,
                match=models.MatchAny(any=[user_value])
            )

        elif right.startswith("document."):
            doc_field = right[9:]
            literal_value = parse_literal_value(left)

            return models.FieldCondition(
                key=doc_field,
                match=models.MatchAny(any=[literal_value])
            )

        elif left.startswith("document."):
            doc_field = left[9:]
            list_values = parse_list_literal(right)

            if list_values is not None:
                if len(list_values) == 0:
                    return models.FieldCondition(
                        key=DENY_ALL_FIELD,
                        match=models.MatchValue(value=DENY_ALL_VALUE)
                    )
                else:
                    return models.FieldCondition(
                        key=doc_field,
                        match=models.MatchAny(any=list_values)
                    )

    return None
