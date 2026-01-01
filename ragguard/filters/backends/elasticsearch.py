"""
Elasticsearch / OpenSearch filter builder.

Converts RAGGuard policies to Elasticsearch/OpenSearch-native filter format.
"""

from typing import TYPE_CHECKING, Any, Optional, Union

from ...exceptions import UnsupportedConditionError
from ...policy.models import Policy, Rule
from ..base import (
    DENY_ALL_FIELD,
    DENY_ALL_VALUE,
    user_satisfies_allow,
    validate_field_path,
)

if TYPE_CHECKING:
    from ...policy.compiler import CompiledCondition, CompiledExpression


def to_elasticsearch_filter(policy: Policy, user: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Convert a policy and user context to an Elasticsearch/OpenSearch filter.

    Both Elasticsearch and OpenSearch use the same Query DSL format with bool queries.
    This function generates native Elasticsearch filters for optimal performance.

    Args:
        policy: The access control policy
        user: User context

    Returns:
        Elasticsearch bool query dict, or None if no filter needed

    Example:
        Input: user.department == document.department
        Output: {"term": {"department": "engineering"}}

        Input: (user.dept == doc.dept OR doc.visibility == 'public')
        Output: {
            "bool": {
                "should": [
                    {"term": {"department": "engineering"}},
                    {"term": {"visibility": "public"}}
                ]
            }
        }

    Elasticsearch Query DSL:
        - term: Exact value match
        - terms: Match any value in list
        - range: Numeric/date comparisons (gt, gte, lt, lte)
        - bool: Combine queries with must (AND), should (OR), must_not (NOT)
        - exists: Check field existence
    """
    should_filters = []  # OR between rules

    for rule in policy.rules:
        rule_filter = _build_elasticsearch_rule_filter(rule, user)
        if rule_filter:
            should_filters.append(rule_filter)

    if not should_filters:
        # No rules apply to this user
        if policy.default == "allow":
            return None  # No filter needed (allow all)
        else:
            # Deny all - return filter that matches nothing
            return {
                "bool": {
                    "must": [
                        {"term": {DENY_ALL_FIELD: DENY_ALL_VALUE}}
                    ]
                }
            }

    if len(should_filters) == 1:
        return should_filters[0]

    # Multiple rules: OR them together
    return {"bool": {"should": should_filters, "minimum_should_match": 1}}


def _build_elasticsearch_rule_filter(
    rule: Rule,
    user: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """Build an Elasticsearch filter for a single rule."""

    # Check if user satisfies the allow conditions
    if not user_satisfies_allow(rule.allow, user):
        return None

    # User is allowed - build filter for document match conditions
    must_filters = []

    # Add match conditions (what documents this rule applies to)
    if rule.match:
        for key, value in rule.match.items():
            if isinstance(value, list):
                # Match any value in list - use "terms" query
                must_filters.append({"terms": {key: value}})
            else:
                # Single value - use "term" query
                must_filters.append({"term": {key: value}})

    # Add conditions that reference user context
    if rule.allow.conditions:
        for condition_str in rule.allow.conditions:
            # Compile the condition and build filter
            from ...policy.compiler import ConditionCompiler

            try:
                compiled = ConditionCompiler.compile_expression(condition_str)
                condition_filter = _build_elasticsearch_from_compiled_node(compiled, user)
                if condition_filter:
                    must_filters.append(condition_filter)
            except (ValueError, AttributeError, KeyError, TypeError):
                # If condition can't be converted to native filter, skip it
                # (This is a limitation - some conditions can't be pre-filtered)
                # Only catch specific exceptions, not system exceptions
                pass

    if not must_filters:
        # No match conditions - this rule applies to all documents for this user
        return {"match_all": {}}

    if len(must_filters) == 1:
        return must_filters[0]

    # Multiple conditions: AND them together
    return {"bool": {"must": must_filters}}


def _build_elasticsearch_from_compiled_node(
    node: Union['CompiledCondition', 'CompiledExpression'],
    user: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """
    Build an Elasticsearch filter from a compiled expression node.

    This enables native OR/AND support by converting the expression tree
    to Elasticsearch's bool query format.

    Args:
        node: CompiledCondition or CompiledExpression from compiler
        user: User context

    Returns:
        Elasticsearch query dict with native OR/AND support
    """
    from ...policy.compiler import CompiledCondition, CompiledExpression, LogicalOperator

    if isinstance(node, CompiledCondition):
        # Convert single condition to Elasticsearch filter
        return _build_elasticsearch_from_condition(node, user)

    elif isinstance(node, CompiledExpression):
        # Convert OR/AND expression tree
        child_filters = []

        for child in node.children:
            child_filter = _build_elasticsearch_from_compiled_node(child, user)
            if child_filter is not None:
                child_filters.append(child_filter)

        if not child_filters:
            return None

        if len(child_filters) == 1:
            return child_filters[0]

        # Build native OR/AND filter
        if node.operator == LogicalOperator.OR:
            # OR: Use 'should' clause (at least one must match)
            return {"bool": {"should": child_filters, "minimum_should_match": 1}}
        elif node.operator == LogicalOperator.AND:
            # AND: Use 'must' clause (all must match)
            return {"bool": {"must": child_filters}}

    return None


def _build_elasticsearch_from_condition(
    condition: 'CompiledCondition',
    user: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """
    Build an Elasticsearch filter from a CompiledCondition.

    Converts the compiled AST node to native Elasticsearch query format.
    """
    from ...policy.compiler import (
        CompiledConditionEvaluator,
        ConditionOperator,
        ConditionType,
        ValueType,
    )

    # Resolve user values at filter build time
    left_value = None
    if condition.left.value_type == ValueType.USER_FIELD:
        left_value = CompiledConditionEvaluator._get_nested_value(user, condition.left.field_path)

    right_value = None
    if condition.right and condition.right.value_type == ValueType.USER_FIELD:
        right_value = CompiledConditionEvaluator._get_nested_value(user, condition.right.field_path)
    elif condition.right and condition.right.value_type in (
        ValueType.LITERAL_STRING,
        ValueType.LITERAL_NUMBER,
        ValueType.LITERAL_BOOL,
        ValueType.LITERAL_LIST
    ):
        right_value = condition.right.value

    # Convert condition to Elasticsearch filter based on operator
    if condition.operator == ConditionOperator.EQUALS:
        # user.field == document.field OR document.field == user.field
        if condition.left.value_type == ValueType.USER_FIELD and \
           condition.right and condition.right.value_type == ValueType.DOCUMENT_FIELD:
            # user.field == document.field
            doc_field = validate_field_path(condition.right.field_path, "elasticsearch")
            return {"term": {doc_field: left_value}}
        elif condition.left.value_type == ValueType.DOCUMENT_FIELD and \
             condition.right and condition.right.value_type == ValueType.USER_FIELD:
            # document.field == user.field
            doc_field = validate_field_path(condition.left.field_path, "elasticsearch")
            return {"term": {doc_field: right_value}}
        elif condition.left.value_type == ValueType.DOCUMENT_FIELD and \
             condition.right and condition.right.value_type in (
                 ValueType.LITERAL_STRING,
                 ValueType.LITERAL_NUMBER,
                 ValueType.LITERAL_BOOL
             ):
            # document.field == literal
            doc_field = validate_field_path(condition.left.field_path, "elasticsearch")
            return {"term": {doc_field: right_value}}

    elif condition.operator == ConditionOperator.NOT_EQUALS:
        # user.field != document.field
        if condition.left.value_type == ValueType.USER_FIELD and \
           condition.right and condition.right.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = validate_field_path(condition.right.field_path, "elasticsearch")
            return {"bool": {"must_not": [{"term": {doc_field: left_value}}]}}
        elif (condition.left.value_type == ValueType.DOCUMENT_FIELD and \
             condition.right and condition.right.value_type == ValueType.USER_FIELD) or (condition.left.value_type == ValueType.DOCUMENT_FIELD and \
             condition.right and condition.right.value_type in (
                 ValueType.LITERAL_STRING,
                 ValueType.LITERAL_NUMBER,
                 ValueType.LITERAL_BOOL
             )):
            doc_field = validate_field_path(condition.left.field_path, "elasticsearch")
            return {"bool": {"must_not": [{"term": {doc_field: right_value}}]}}

    elif condition.operator == ConditionOperator.IN:
        # Check condition type
        if condition.condition_type == ConditionType.VALUE_IN_ARRAY_FIELD:
            # user.id in document.allowed_users
            # Elasticsearch doesn't support this directly - can't filter
            return None
        elif condition.left.value_type == ValueType.DOCUMENT_FIELD and \
             condition.right and condition.right.value_type == ValueType.LITERAL_LIST:
            # document.field in ['a', 'b', 'c']
            doc_field = validate_field_path(condition.left.field_path, "elasticsearch")
            return {"terms": {doc_field: right_value}}

    elif condition.operator == ConditionOperator.NOT_IN:
        if condition.condition_type == ConditionType.VALUE_IN_ARRAY_FIELD:
            # user.id not in document.allowed_users
            return None
        elif condition.left.value_type == ValueType.DOCUMENT_FIELD and \
             condition.right and condition.right.value_type == ValueType.LITERAL_LIST:
            # document.field not in ['a', 'b', 'c']
            doc_field = validate_field_path(condition.left.field_path, "elasticsearch")
            return {"bool": {"must_not": [{"terms": {doc_field: right_value}}]}}

    elif condition.operator == ConditionOperator.EXISTS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = validate_field_path(condition.left.field_path, "elasticsearch")
            return {"exists": {"field": doc_field}}

    elif condition.operator == ConditionOperator.NOT_EXISTS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = validate_field_path(condition.left.field_path, "elasticsearch")
            return {"bool": {"must_not": [{"exists": {"field": doc_field}}]}}

    elif condition.operator == ConditionOperator.GREATER_THAN:
        # document.field > user.field OR document.field > literal
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = validate_field_path(condition.left.field_path, "elasticsearch")
            value = right_value if condition.right else None
            if value is not None:
                return {"range": {doc_field: {"gt": value}}}

    elif condition.operator == ConditionOperator.GREATER_THAN_OR_EQUAL:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = validate_field_path(condition.left.field_path, "elasticsearch")
            value = right_value if condition.right else None
            if value is not None:
                return {"range": {doc_field: {"gte": value}}}

    elif condition.operator == ConditionOperator.LESS_THAN:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = validate_field_path(condition.left.field_path, "elasticsearch")
            value = right_value if condition.right else None
            if value is not None:
                return {"range": {doc_field: {"lt": value}}}

    elif condition.operator == ConditionOperator.LESS_THAN_OR_EQUAL:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = validate_field_path(condition.left.field_path, "elasticsearch")
            value = right_value if condition.right else None
            if value is not None:
                return {"range": {doc_field: {"lte": value}}}

    # If we reach here with an unhandled operator, raise an error instead of silently allowing
    raise UnsupportedConditionError(
        condition=str(condition.operator),
        backend="elasticsearch",
        reason=f"Operator {condition.operator} with value types "
               f"left={condition.left.value_type}, right={condition.right.value_type if condition.right else None} "
               f"is not supported. This is a security measure to prevent silent filter bypass."
    )
