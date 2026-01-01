"""
ChromaDB filter builder.

Converts RAGGuard policies to ChromaDB where filters.
ChromaDB uses dict-based filters with $eq, $in, $and, $or operators (similar to Pinecone).
"""

import re
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


def to_chromadb_filter(policy: Policy, user: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Convert a policy and user context to a ChromaDB where filter.

    ChromaDB uses dict-based filters with $eq, $in, $and, $or operators (similar to Pinecone).

    Args:
        policy: The access control policy
        user: User context

    Returns:
        ChromaDB filter dict or None if no filtering needed

    Example:
        {
            "$or": [
                {"department": {"$eq": "engineering"}},
                {"public": {"$eq": True}}
            ]
        }
    """
    or_filters = []  # OR between rules

    for rule in policy.rules:
        rule_filter = _build_chromadb_rule_filter(rule, user)
        if rule_filter is SKIP_RULE:
            # User doesn't satisfy this rule's allow conditions - skip to next
            continue
        if rule_filter is None:
            # This rule grants access to all documents
            # No need to check other rules - return None (no filter)
            return None
        or_filters.append(rule_filter)

    if not or_filters:
        # No rules apply to this user
        if policy.default == "allow":
            # Allow all - return None (no filter)
            return None
        else:
            # Deny all - return filter that matches nothing
            return {DENY_ALL_FIELD: {"$eq": DENY_ALL_VALUE}}

    if len(or_filters) == 1:
        return or_filters[0]

    # Multiple rules: OR them together
    return {"$or": or_filters}


def _build_chromadb_rule_filter(
    rule: Rule,
    user: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """Build a ChromaDB filter for a single rule.

    Returns:
        - dict: A filter for this rule
        - None: Rule matches all documents (no restrictions)
        - SKIP_RULE: User doesn't satisfy this rule's allow conditions
    """
    # Check if user satisfies the allow conditions
    if not user_satisfies_allow(rule.allow, user):
        return SKIP_RULE

    # User is allowed - build filter for document match conditions
    and_filters = []

    # Add match conditions (what documents this rule applies to)
    if rule.match:
        for field, value in rule.match.items():
            if isinstance(value, list):
                # Multiple values: use $in operator
                and_filters.append({field: {"$in": value}})
            else:
                # Single value: use $eq operator
                and_filters.append({field: {"$eq": value}})

    # Add dynamic conditions (runtime comparisons like user.dept == doc.dept)
    if rule.allow.conditions:
        for condition in rule.allow.conditions:
            cond_filter = _build_chromadb_condition_filter(condition, user)
            if cond_filter:
                and_filters.append(cond_filter)

    # If no conditions at all, this rule matches everything
    if not and_filters:
        # No restrictions for this rule
        return None

    if len(and_filters) == 1:
        return and_filters[0]

    # Multiple conditions: AND them together
    return {"$and": and_filters}


def _build_chromadb_condition_filter(
    condition: str,
    user: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """
    Build a ChromaDB filter from a condition expression.

    Handles conditions like:
    - user.department == document.department
    - user.id in document.shared_with
    - document.access_level == 'public'
    - document.access_level != 'restricted'
    - document.reviewed_at exists
    - document.draft_notes not exists
    - (document.status == 'published' OR document.reviewed == true)  # Native OR
    """
    condition = condition.strip()

    # NEW: Check if this is an OR/AND expression
    # If it contains OR/AND operators (case-insensitive, whole word), use compiled expression builder
    if re.search(r'\b(OR|AND)\b', condition, re.IGNORECASE):
        # This is a complex expression with OR/AND - use new logic
        from ...policy.compiler import CompiledExpression, ConditionCompiler

        try:
            compiled = ConditionCompiler.compile_expression(condition)

            # If it's a CompiledExpression (has OR/AND), use new native filter builder
            if isinstance(compiled, CompiledExpression):
                return _build_chromadb_from_compiled_node(compiled, user)

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
            # ChromaDB: Field not exists - check if field is None/null
            return {doc_field: {"$eq": None}}
        return None

    elif " exists" in condition:
        field = condition.replace(" exists", "").strip()
        if field.startswith("document."):
            doc_field = field[9:]
            # ChromaDB: Field exists - check if field is not None/null
            return {doc_field: {"$ne": None}}
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

            # ChromaDB: Use $ne operator
            return {doc_field: {"$ne": literal_value}}

    # Parse user.field == document.field or document.field == 'literal'
    if "==" in condition:
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
                return {DENY_ALL_FIELD: {"$eq": DENY_ALL_VALUE}}

            # Check if right side is document field
            if right.startswith("document."):
                doc_field = right[9:]
                return {doc_field: {"$eq": user_value}}

        # Handle document.field == 'literal'
        elif left.startswith("document."):
            doc_field = left[9:]
            literal_value = parse_literal_value(right)
            return {doc_field: {"$eq": literal_value}}

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

            # ChromaDB: Use $nin operator to exclude the user value from array
            return {doc_field: {"$nin": [user_value]}}

        # Check if left is literal value and right is document array field
        elif right.startswith("document."):
            doc_field = right[9:]
            literal_value = parse_literal_value(left)

            # ChromaDB: Use $nin operator to exclude literal from array
            return {doc_field: {"$nin": [literal_value]}}

        # Check if left is document field and right is list literal
        elif left.startswith("document."):
            doc_field = left[9:]
            list_values = parse_list_literal(right)

            if list_values is not None:
                if len(list_values) == 0:
                    # Empty list: NOT IN [] means everything is allowed
                    # Return None (no filter restriction)
                    return None
                else:
                    # ChromaDB: Use $nin operator to exclude values
                    return {doc_field: {"$nin": list_values}}

    # Parse user.field in document.field OR document.field in ['a', 'b', 'c'] OR 'literal' in document.array
    elif " in " in condition and " not in " not in condition:
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
                return {DENY_ALL_FIELD: {"$eq": DENY_ALL_VALUE}}

            # ChromaDB uses $in operator with array field containing value
            # This checks if array field contains the user value
            return {doc_field: {"$in": [user_value]}}

        # Check if left is literal value and right is document array field
        elif right.startswith("document."):
            doc_field = right[9:]
            literal_value = parse_literal_value(left)

            # ChromaDB: Use $in operator to check if array contains literal
            return {doc_field: {"$in": [literal_value]}}

        # Check if left is document field and right is list literal
        elif left.startswith("document."):
            doc_field = left[9:]
            list_values = parse_list_literal(right)

            if list_values is not None:
                if len(list_values) == 0:
                    # Empty list should match nothing - create impossible filter
                    return {DENY_ALL_FIELD: {"$eq": DENY_ALL_VALUE}}
                else:
                    # ChromaDB: Use $in operator to match any value in the list
                    return {doc_field: {"$in": list_values}}

    return None


def _build_chromadb_from_condition(
    condition: 'CompiledCondition',
    user: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """
    Build a ChromaDB filter from a CompiledCondition node.

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
            return {doc_field: {"$ne": None}}

    elif condition.operator == ConditionOperator.NOT_EXISTS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = ".".join(condition.left.field_path)
            return {doc_field: {"$eq": None}}

    # Handle EQUALS operator
    elif condition.operator == ConditionOperator.EQUALS:
        # user.field == document.field
        if condition.left.value_type == ValueType.USER_FIELD and condition.right.value_type == ValueType.DOCUMENT_FIELD:
            if left_value is None:
                return {DENY_ALL_FIELD: {"$eq": DENY_ALL_VALUE}}
            doc_field = ".".join(condition.right.field_path)
            return {doc_field: {"$eq": left_value}}

        # document.field == user.field (reversed)
        elif condition.left.value_type == ValueType.DOCUMENT_FIELD and condition.right.value_type == ValueType.USER_FIELD:
            if right_value is None:
                return {DENY_ALL_FIELD: {"$eq": DENY_ALL_VALUE}}
            doc_field = ".".join(condition.left.field_path)
            return {doc_field: {"$eq": right_value}}

        # document.field == literal
        elif condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = ".".join(condition.left.field_path)
            return {doc_field: {"$eq": right_value}}

    # Handle NOT_EQUALS operator
    elif condition.operator == ConditionOperator.NOT_EQUALS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = ".".join(condition.left.field_path)
            return {doc_field: {"$ne": right_value}}

    # Handle IN operator
    elif condition.operator == ConditionOperator.IN:
        # user.id in document.array_field
        if condition.left.value_type == ValueType.USER_FIELD and condition.right.value_type == ValueType.DOCUMENT_FIELD:
            if left_value is None:
                return {DENY_ALL_FIELD: {"$eq": DENY_ALL_VALUE}}
            doc_field = ".".join(condition.right.field_path)
            return {doc_field: {"$in": [left_value]}}

        # document.field in [literal values]
        elif condition.left.value_type == ValueType.DOCUMENT_FIELD and isinstance(right_value, list):
            doc_field = ".".join(condition.left.field_path)
            if len(right_value) == 0:
                return {DENY_ALL_FIELD: {"$eq": DENY_ALL_VALUE}}
            return {doc_field: {"$in": right_value}}

    # Handle NOT_IN operator
    elif condition.operator == ConditionOperator.NOT_IN:
        # user.id not in document.array_field
        if condition.left.value_type == ValueType.USER_FIELD and condition.right.value_type == ValueType.DOCUMENT_FIELD:
            if left_value is None:
                return None  # No restriction
            doc_field = ".".join(condition.right.field_path)
            return {doc_field: {"$nin": [left_value]}}

        # document.field not in [literal values]
        elif condition.left.value_type == ValueType.DOCUMENT_FIELD and isinstance(right_value, list):
            doc_field = ".".join(condition.left.field_path)
            if len(right_value) == 0:
                return None  # No restriction
            return {doc_field: {"$nin": right_value}}

    # Handle comparison operators
    elif condition.operator == ConditionOperator.LESS_THAN:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = ".".join(condition.left.field_path)
            return {doc_field: {"$lt": right_value}}

    elif condition.operator == ConditionOperator.LESS_THAN_OR_EQUAL:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = ".".join(condition.left.field_path)
            return {doc_field: {"$lte": right_value}}

    elif condition.operator == ConditionOperator.GREATER_THAN:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = ".".join(condition.left.field_path)
            return {doc_field: {"$gt": right_value}}

    elif condition.operator == ConditionOperator.GREATER_THAN_OR_EQUAL:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = ".".join(condition.left.field_path)
            return {doc_field: {"$gte": right_value}}

    # If we reach here with an unhandled operator, raise an error instead of silently allowing
    raise UnsupportedConditionError(
        condition=str(condition.operator),
        backend="chromadb",
        reason=f"Operator {condition.operator} with value types "
               f"left={condition.left.value_type}, right={condition.right.value_type if condition.right else None} "
               f"is not supported. This is a security measure to prevent silent filter bypass."
    )


def _build_chromadb_from_compiled_node(
    node: Union['CompiledCondition', 'CompiledExpression'],
    user: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """
    Convert a compiled expression tree to a native ChromaDB filter.

    This enables native OR/AND filtering in ChromaDB instead of post-filtering.

    Args:
        node: CompiledCondition or CompiledExpression tree
        user: User context for substitution

    Returns:
        ChromaDB filter dict with $or/$and operators, or None if no filter needed

    Example:
        Input: (document.status == 'published' OR document.reviewed == true)
        Output: {
            "$or": [
                {"status": {"$eq": "published"}},
                {"reviewed": {"$eq": True}}
            ]
        }
    """
    from ...policy.compiler import CompiledCondition, CompiledExpression, LogicalOperator

    if isinstance(node, CompiledCondition):
        # Base case: single condition
        return _build_chromadb_from_condition(node, user)

    elif isinstance(node, CompiledExpression):
        # Recursive case: OR/AND expression
        child_filters = []

        for child in node.children:
            child_filter = _build_chromadb_from_compiled_node(child, user)
            if child_filter is not None:
                child_filters.append(child_filter)

        # If no valid filters, return None
        if not child_filters:
            return None

        # If only one filter, return it directly (no need for $or/$and wrapper)
        if len(child_filters) == 1:
            return child_filters[0]

        # Multiple filters: wrap with $or or $and
        if node.operator == LogicalOperator.OR:
            return {"$or": child_filters}
        elif node.operator == LogicalOperator.AND:
            return {"$and": child_filters}

    return None
