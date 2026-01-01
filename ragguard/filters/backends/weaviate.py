"""
Weaviate filter builder.

Converts RAGGuard policies to Weaviate where filter format.
"""

from typing import TYPE_CHECKING, Any, Optional, Union

from ...exceptions import UnsupportedConditionError
from ...policy.models import Policy, Rule
from ..base import (
    DENY_ALL_FIELD,
    DENY_ALL_VALUE,
    SKIP_RULE,
    get_nested_value,
    parse_list_literal,
    parse_literal_value,
    user_satisfies_allow,
)

if TYPE_CHECKING:
    from ...policy.compiler import CompiledCondition, CompiledExpression


def to_weaviate_filter(policy: Policy, user: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Convert a policy and user context to a Weaviate where filter.

    Weaviate uses a dict-based filter format with operators like Equal, ContainsAny, And, Or.
    The filter uses OR logic between rules and AND logic within a rule.

    Args:
        policy: The access control policy
        user: User context

    Returns:
        Weaviate where filter (dict format) or None if no filtering needed

    Example:
        {
            "operator": "Or",
            "operands": [
                {
                    "path": ["department"],
                    "operator": "Equal",
                    "valueText": "engineering"
                }
            ]
        }
    """
    or_operands = []  # OR between rules

    for rule in policy.rules:
        rule_filter = _build_weaviate_rule_filter(rule, user)
        if rule_filter is SKIP_RULE:
            # User doesn't satisfy this rule's allow conditions - skip to next
            continue
        if rule_filter is None:
            # This rule matches all documents - return None (no filter needed)
            return None
        or_operands.append(rule_filter)

    if not or_operands:
        # No rules apply to this user
        if policy.default == "allow":
            # Allow all - return None (no filter)
            return None
        else:
            # Deny all - return filter that matches nothing
            return {
                "path": [DENY_ALL_FIELD],
                "operator": "Equal",
                "valueText": DENY_ALL_VALUE
            }

    if len(or_operands) == 1:
        return or_operands[0]

    # Multiple rules: OR them together
    return {
        "operator": "Or",
        "operands": or_operands
    }


def _build_weaviate_rule_filter(
    rule: Rule,
    user: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """Build a Weaviate filter for a single rule.

    Returns:
        - dict: A filter for this rule
        - None: Rule matches all documents (no restrictions)
        - SKIP_RULE: User doesn't satisfy this rule's allow conditions
    """
    # Check if user satisfies the allow conditions
    if not user_satisfies_allow(rule.allow, user):
        return SKIP_RULE

    # User is allowed - build filter for document match conditions
    and_operands = []

    # Add match conditions (what documents this rule applies to)
    if rule.match:
        for field, value in rule.match.items():
            if isinstance(value, list):
                # Multiple values: use ContainsAny for OR within field
                and_operands.append({
                    "path": [field],
                    "operator": "ContainsAny",
                    "valueText": value
                })
            else:
                # Single value: use Equal
                value_key = _weaviate_value_key(value)
                and_operands.append({
                    "path": [field],
                    "operator": "Equal",
                    value_key: value
                })

    # Add dynamic conditions (runtime comparisons like user.dept == doc.dept)
    if rule.allow.conditions:
        for condition in rule.allow.conditions:
            cond_filter = _build_weaviate_condition_filter(condition, user)
            if cond_filter:
                and_operands.append(cond_filter)

    # If no conditions at all, this rule matches everything
    if not and_operands:
        # Return a filter that matches all documents (no restrictions)
        # In Weaviate, we can't easily express "match all", so we return None
        # which will be handled by the caller
        return None

    if len(and_operands) == 1:
        return and_operands[0]

    # Multiple conditions: AND them together
    return {
        "operator": "And",
        "operands": and_operands
    }


def _build_weaviate_condition_filter(
    condition: str,
    user: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """
    Build a Weaviate filter from a condition expression.

    Handles conditions like:
    - user.department == document.department
    - user.id in document.shared_with
    - document.access_level == 'public'
    - document.level != 'restricted'
    - document.reviewed_at exists
    - document.draft_notes not exists
    - (document.status == 'published' OR document.reviewed == true)  # Native OR
    """
    condition = condition.strip()

    # NEW: Check if this is an OR/AND expression
    # If it contains OR/AND operators (case-insensitive, whole word), use compiled expression builder
    import re
    if re.search(r'\b(OR|AND)\b', condition, re.IGNORECASE):
        # This is a complex expression with OR/AND - use new logic
        from ...policy.compiler import CompiledExpression, ConditionCompiler

        try:
            compiled = ConditionCompiler.compile_expression(condition)

            # If it's a CompiledExpression (has OR/AND), use new native filter builder
            if isinstance(compiled, CompiledExpression):
                return _build_weaviate_from_compiled_node(compiled, user)

            # Otherwise fall through to existing string parsing (backward compat)
        except (ValueError, AttributeError, KeyError, TypeError):
            # Fallback to existing string parsing if compilation fails
            # Only catch specific exceptions, not system exceptions
            pass

    # Parse field existence checks (order matters - check "not exists" before "!=")
    if " not exists" in condition:
        field = condition.replace(" not exists", "").strip()
        if field.startswith("document."):
            doc_field = field[9:]
            # Weaviate: Use IsNull operator for non-existence
            return {
                "path": [doc_field],
                "operator": "IsNull",
                "valueBoolean": True
            }
        return None

    elif " exists" in condition:
        field = condition.replace(" exists", "").strip()
        if field.startswith("document."):
            doc_field = field[9:]
            # Weaviate: Use IsNull with False for existence
            return {
                "path": [doc_field],
                "operator": "IsNull",
                "valueBoolean": False
            }
        return None

    # Parse document.field != 'literal' (negation)
    elif "!=" in condition:
        parts = condition.split("!=", 1)
        if len(parts) != 2:
            return None

        left, right = parts[0].strip(), parts[1].strip()

        # Handle document.field != 'literal'
        if left.startswith("document."):
            doc_field = left[9:]
            literal_value = parse_literal_value(right)
            value_key = _weaviate_value_key(literal_value)

            # Weaviate: Use NotEqual operator
            return {
                "path": [doc_field],
                "operator": "NotEqual",
                value_key: literal_value
            }

    # Parse user.field == document.field or document.field == 'literal'
    elif "==" in condition:
        parts = condition.split("==", 1)
        if len(parts) != 2:
            return None

        left, right = parts[0].strip(), parts[1].strip()

        # Check if left side is user context
        if left.startswith("user."):
            user_field = left[5:]
            user_value = get_nested_value(user, user_field)

            # If user field is None/missing, create an impossible filter (deny all)
            if user_value is None:
                return {
                    "path": [DENY_ALL_FIELD],
                    "operator": "Equal",
                    "valueText": DENY_ALL_VALUE
                }

            # Check if right side is document field
            if right.startswith("document."):
                doc_field = right[9:]
                value_key = _weaviate_value_key(user_value)
                return {
                    "path": [doc_field],
                    "operator": "Equal",
                    value_key: user_value
                }

        # Handle document.field == 'literal'
        elif left.startswith("document."):
            doc_field = left[9:]
            literal_value = parse_literal_value(right)
            value_key = _weaviate_value_key(literal_value)
            return {
                "path": [doc_field],
                "operator": "Equal",
                value_key: literal_value
            }

    # Parse document.field not in ['a', 'b', 'c'] OR user.field not in document.array OR 'literal' not in document.array
    elif " not in " in condition:
        parts = condition.split(" not in ", 1)
        if len(parts) != 2:
            return None

        left, right = parts[0].strip(), parts[1].strip()

        # Check if left is user context and right is document array field
        if left.startswith("user.") and right.startswith("document."):
            user_field = left[5:]
            doc_field = right[9:]
            user_value = get_nested_value(user, user_field)

            # If user field is None/missing, condition is satisfied (user not in list)
            if user_value is None:
                return None  # No filter restriction

            # Weaviate: Use Not with ContainsAny
            return {
                "operator": "Not",
                "operands": [{
                    "path": [doc_field],
                    "operator": "ContainsAny",
                    "valueText": [user_value]
                }]
            }

        # Check if left is literal value and right is document array field
        elif right.startswith("document."):
            doc_field = right[9:]
            literal_value = parse_literal_value(left)
            value_key = _weaviate_value_key(literal_value)

            # Weaviate: Use Not with ContainsAny for literal
            return {
                "operator": "Not",
                "operands": [{
                    "path": [doc_field],
                    "operator": "ContainsAny",
                    value_key: [literal_value]
                }]
            }

        # Check if left is document field and right is list literal
        elif left.startswith("document."):
            doc_field = left[9:]
            list_values = parse_list_literal(right)

            if list_values is not None:
                if len(list_values) == 0:
                    # Empty list: NOT IN [] means everything is allowed
                    # Return None (no filter = allow all)
                    return None
                else:
                    # Weaviate: Use Not with ContainsAny
                    value_key = _weaviate_value_key(list_values[0])
                    return {
                        "operator": "Not",
                        "operands": [{
                            "path": [doc_field],
                            "operator": "ContainsAny",
                            value_key: list_values
                        }]
                    }

    # Parse user.field in document.field OR document.field in ['a', 'b', 'c'] OR 'literal' in document.array
    elif " in " in condition:
        parts = condition.split(" in ")
        if len(parts) != 2:
            return None

        left, right = parts[0].strip(), parts[1].strip()

        # Check if left is user context and right is document field
        if left.startswith("user.") and right.startswith("document."):
            user_field = left[5:]
            doc_field = right[9:]
            user_value = get_nested_value(user, user_field)

            # If user field is None/missing, create an impossible filter (deny all)
            if user_value is None:
                return {
                    "path": [DENY_ALL_FIELD],
                    "operator": "Equal",
                    "valueText": DENY_ALL_VALUE
                }

            # ContainsAny checks if array field contains the user value
            return {
                "path": [doc_field],
                "operator": "ContainsAny",
                "valueText": [user_value]
            }

        # Check if left is literal value and right is document array field
        elif right.startswith("document."):
            doc_field = right[9:]
            literal_value = parse_literal_value(left)
            value_key = _weaviate_value_key(literal_value)

            # ContainsAny checks if array field contains the literal value
            return {
                "path": [doc_field],
                "operator": "ContainsAny",
                value_key: [literal_value]
            }

        # Check if left is document field and right is list literal
        elif left.startswith("document."):
            doc_field = left[9:]
            list_values = parse_list_literal(right)

            if list_values is not None:
                if len(list_values) == 0:
                    # Empty list: IN [] should match nothing
                    return {
                        "path": [DENY_ALL_FIELD],
                        "operator": "Equal",
                        "valueText": DENY_ALL_VALUE
                    }
                else:
                    # Weaviate: Use ContainsAny to match any value in the list
                    # Determine the value key based on the first element type
                    value_key = _weaviate_value_key(list_values[0])
                    return {
                        "path": [doc_field],
                        "operator": "ContainsAny",
                        value_key: list_values
                    }

    # For legacy string parsing, return None for unhandled conditions
    # The compiled condition handler (below) handles errors for CompiledCondition objects
    return None


def _build_weaviate_from_condition(
    condition: 'CompiledCondition',
    user: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """
    Build a Weaviate filter from a CompiledCondition node.

    This function handles individual conditions that have been parsed into AST form.
    Supports all operators: ==, !=, in, not in, exists, not exists, <, >, <=, >=
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
            return {
                "path": [doc_field],
                "operator": "IsNull",
                "valueBoolean": False
            }

    elif condition.operator == ConditionOperator.NOT_EXISTS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = ".".join(condition.left.field_path)
            return {
                "path": [doc_field],
                "operator": "IsNull",
                "valueBoolean": True
            }

    # Handle EQUALS operator
    elif condition.operator == ConditionOperator.EQUALS:
        # user.field == document.field
        if condition.left.value_type == ValueType.USER_FIELD and condition.right.value_type == ValueType.DOCUMENT_FIELD:
            if left_value is None:
                return {
                    "path": [DENY_ALL_FIELD],
                    "operator": "Equal",
                    "valueText": DENY_ALL_VALUE
                }
            doc_field = ".".join(condition.right.field_path)
            value_key = _weaviate_value_key(left_value)
            return {
                "path": [doc_field],
                "operator": "Equal",
                value_key: left_value
            }

        # document.field == user.field (reversed)
        elif condition.left.value_type == ValueType.DOCUMENT_FIELD and condition.right.value_type == ValueType.USER_FIELD:
            if right_value is None:
                return {
                    "path": [DENY_ALL_FIELD],
                    "operator": "Equal",
                    "valueText": DENY_ALL_VALUE
                }
            doc_field = ".".join(condition.left.field_path)
            value_key = _weaviate_value_key(right_value)
            return {
                "path": [doc_field],
                "operator": "Equal",
                value_key: right_value
            }

        # document.field == literal
        elif condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = ".".join(condition.left.field_path)
            value_key = _weaviate_value_key(right_value)
            return {
                "path": [doc_field],
                "operator": "Equal",
                value_key: right_value
            }

    # Handle NOT_EQUALS operator
    elif condition.operator == ConditionOperator.NOT_EQUALS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = ".".join(condition.left.field_path)
            value_key = _weaviate_value_key(right_value)
            return {
                "path": [doc_field],
                "operator": "NotEqual",
                value_key: right_value
            }

    # Handle IN operator
    elif condition.operator == ConditionOperator.IN:
        # user.id in document.array_field
        if condition.left.value_type == ValueType.USER_FIELD and condition.right.value_type == ValueType.DOCUMENT_FIELD:
            if left_value is None:
                return {
                    "path": [DENY_ALL_FIELD],
                    "operator": "Equal",
                    "valueText": DENY_ALL_VALUE
                }
            doc_field = ".".join(condition.right.field_path)
            value_key = _weaviate_value_key(left_value)
            return {
                "path": [doc_field],
                "operator": "ContainsAny",
                value_key: [left_value]
            }

        # document.field in [literal values]
        elif condition.left.value_type == ValueType.DOCUMENT_FIELD and isinstance(right_value, list):
            doc_field = ".".join(condition.left.field_path)
            if len(right_value) == 0:
                return {
                    "path": [DENY_ALL_FIELD],
                    "operator": "Equal",
                    "valueText": DENY_ALL_VALUE
                }
            value_key = _weaviate_value_key(right_value[0]) if right_value else "valueText"
            return {
                "path": [doc_field],
                "operator": "ContainsAny",
                value_key: right_value
            }

    # Handle NOT_IN operator
    elif condition.operator == ConditionOperator.NOT_IN:
        # user.id not in document.array_field
        if condition.left.value_type == ValueType.USER_FIELD and condition.right.value_type == ValueType.DOCUMENT_FIELD:
            if left_value is None:
                return None  # No restriction
            doc_field = ".".join(condition.right.field_path)
            value_key = _weaviate_value_key(left_value)
            return {
                "operator": "Not",
                "operands": [{
                    "path": [doc_field],
                    "operator": "ContainsAny",
                    value_key: [left_value]
                }]
            }

        # document.field not in [literal values]
        elif condition.left.value_type == ValueType.DOCUMENT_FIELD and isinstance(right_value, list):
            doc_field = ".".join(condition.left.field_path)
            if len(right_value) == 0:
                return None  # No restriction
            value_key = _weaviate_value_key(right_value[0]) if right_value else "valueText"
            return {
                "operator": "Not",
                "operands": [{
                    "path": [doc_field],
                    "operator": "ContainsAny",
                    value_key: right_value
                }]
            }

    # Handle comparison operators
    elif condition.operator == ConditionOperator.LESS_THAN:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = ".".join(condition.left.field_path)
            value_key = _weaviate_value_key(right_value)
            return {
                "path": [doc_field],
                "operator": "LessThan",
                value_key: right_value
            }

    elif condition.operator == ConditionOperator.LESS_THAN_OR_EQUAL:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = ".".join(condition.left.field_path)
            value_key = _weaviate_value_key(right_value)
            return {
                "path": [doc_field],
                "operator": "LessThanEqual",
                value_key: right_value
            }

    elif condition.operator == ConditionOperator.GREATER_THAN:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = ".".join(condition.left.field_path)
            value_key = _weaviate_value_key(right_value)
            return {
                "path": [doc_field],
                "operator": "GreaterThan",
                value_key: right_value
            }

    elif condition.operator == ConditionOperator.GREATER_THAN_OR_EQUAL:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = ".".join(condition.left.field_path)
            value_key = _weaviate_value_key(right_value)
            return {
                "path": [doc_field],
                "operator": "GreaterThanEqual",
                value_key: right_value
            }

        # user.field >= literal (e.g., user.level >= 5)
        # This is evaluated at filter-build time, not as a database filter
        elif condition.left.value_type == ValueType.USER_FIELD and condition.right.value_type == ValueType.LITERAL_NUMBER:
            comparison_result = False
            if left_value is not None and right_value is not None:
                comparison_result = left_value >= right_value

            if comparison_result:
                return None  # Condition is true - no filter needed
            else:
                return {
                    "path": [DENY_ALL_FIELD],
                    "operator": "Equal",
                    "valueText": DENY_ALL_VALUE
                }

    # Handle user.field == literal cases not yet covered above
    if condition.operator == ConditionOperator.EQUALS:
        if condition.left.value_type == ValueType.USER_FIELD and condition.right.value_type in (ValueType.LITERAL_STRING, ValueType.LITERAL_NUMBER, ValueType.LITERAL_BOOL):
            if left_value == right_value:
                return None  # Condition is true - no filter needed
            else:
                return {
                    "path": [DENY_ALL_FIELD],
                    "operator": "Equal",
                    "valueText": DENY_ALL_VALUE
                }

    # Handle user.field vs literal comparison operators
    if condition.operator in (ConditionOperator.GREATER_THAN, ConditionOperator.LESS_THAN, ConditionOperator.LESS_THAN_OR_EQUAL):
        if condition.left.value_type == ValueType.USER_FIELD and condition.right.value_type == ValueType.LITERAL_NUMBER:
            comparison_result = False
            if left_value is not None and right_value is not None:
                if condition.operator == ConditionOperator.GREATER_THAN:
                    comparison_result = left_value > right_value
                elif condition.operator == ConditionOperator.LESS_THAN:
                    comparison_result = left_value < right_value
                elif condition.operator == ConditionOperator.LESS_THAN_OR_EQUAL:
                    comparison_result = left_value <= right_value

            if comparison_result:
                return None  # Condition is true - no filter needed
            else:
                return {
                    "path": [DENY_ALL_FIELD],
                    "operator": "Equal",
                    "valueText": DENY_ALL_VALUE
                }

    # If we reach here with an unhandled operator, raise an error instead of silently allowing
    raise UnsupportedConditionError(
        condition=str(condition.operator),
        backend="weaviate",
        reason=f"Operator {condition.operator} with value types "
               f"left={condition.left.value_type}, right={condition.right.value_type if condition.right else None} "
               f"is not supported. This is a security measure to prevent silent filter bypass."
    )


def _build_weaviate_from_compiled_node(
    node: Union['CompiledCondition', 'CompiledExpression'],
    user: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """
    Convert a compiled expression tree to a native Weaviate filter.

    This enables native OR/AND filtering in Weaviate instead of post-filtering.

    Args:
        node: CompiledCondition or CompiledExpression tree
        user: User context for substitution

    Returns:
        Weaviate filter dict with Or/And operators, or None if no filter needed

    Example:
        Input: (document.status == 'published' OR document.reviewed == true)
        Output: {
            "operator": "Or",
            "operands": [
                {"path": ["status"], "operator": "Equal", "valueText": "published"},
                {"path": ["reviewed"], "operator": "Equal", "valueBoolean": True}
            ]
        }
    """
    from ...policy.compiler import CompiledCondition, CompiledExpression, LogicalOperator

    if isinstance(node, CompiledCondition):
        # Base case: single condition
        return _build_weaviate_from_condition(node, user)

    elif isinstance(node, CompiledExpression):
        # Recursive case: OR/AND expression
        child_filters = []

        for child in node.children:
            child_filter = _build_weaviate_from_compiled_node(child, user)
            if child_filter is not None:
                child_filters.append(child_filter)

        # If no valid filters, return None
        if not child_filters:
            return None

        # If only one filter, return it directly (no need for Or/And wrapper)
        if len(child_filters) == 1:
            return child_filters[0]

        # Multiple filters: wrap with Or or And
        if node.operator == LogicalOperator.OR:
            return {
                "operator": "Or",
                "operands": child_filters
            }
        elif node.operator == LogicalOperator.AND:
            return {
                "operator": "And",
                "operands": child_filters
            }

    return None


def _weaviate_value_key(value: Any) -> str:
    """
    Determine the correct Weaviate value key based on type.

    Weaviate uses different keys for different types:
    - valueText for strings
    - valueInt for integers
    - valueNumber for floats
    - valueBoolean for booleans
    """
    if isinstance(value, bool):
        return "valueBoolean"
    elif isinstance(value, int):
        return "valueInt"
    elif isinstance(value, float):
        return "valueNumber"
    else:
        return "valueText"
