"""
Azure AI Search filter builder.

Converts RAGGuard policies to Azure AI Search OData filter format.
"""

from typing import TYPE_CHECKING, Any, Optional, Union

from ...exceptions import UnsupportedConditionError
from ...policy.models import Policy, Rule
from ..base import (
    SKIP_RULE,
    user_satisfies_allow,
    validate_field_path,
)


def _validate_azure_field_path(field_path: list[str]) -> str:
    """Validate field path and join with Azure's / separator."""
    # Validate using standard validation
    validate_field_path(field_path, "azure_search")
    # Azure uses / for nested field access
    return "/".join(field_path)

if TYPE_CHECKING:
    from ...policy.compiler import CompiledCondition, CompiledExpression


def to_azure_search_filter(policy: Policy, user: dict[str, Any]) -> Optional[str]:
    """
    Convert a policy and user context to an Azure AI Search OData filter.

    Azure AI Search (formerly Azure Cognitive Search) uses OData v4 filter syntax.
    This function generates native Azure filters for optimal performance.

    Args:
        policy: The access control policy
        user: User context

    Returns:
        OData filter string, or None if no filter needed

    Example:
        Input: user.department == document.department
        Output: "department eq 'engineering'"

        Input: (user.dept == doc.dept OR doc.visibility == 'public')
        Output: "(department eq 'engineering' or visibility eq 'public')"

    Azure AI Search OData Filter Syntax:
        - Equality: eq, ne
        - Comparison: gt, ge, lt, le
        - Logical: and, or, not
        - Grouping: ( )
        - Collections: any(), all()
        - Examples:
          * "department eq 'engineering'"
          * "clearance ge 3 and department eq 'engineering'"
          * "tags/any(t: t eq 'public')"
    """
    or_clauses = []  # OR between rules

    for rule in policy.rules:
        rule_filter = _build_azure_search_rule_filter(rule, user)
        if rule_filter is SKIP_RULE:
            # User doesn't satisfy this rule's allow conditions - skip to next
            continue
        if rule_filter is None:
            # This rule matches all documents - return None (no filter needed)
            return None
        or_clauses.append(rule_filter)

    if not or_clauses:
        # No rules apply to this user
        if policy.default == "allow":
            return None  # No filter needed (allow all)
        else:
            # Deny all - return filter that matches nothing
            return "search.score() eq -1"  # Impossible condition

    if len(or_clauses) == 1:
        return or_clauses[0]

    # Multiple rules: OR them together
    return "(" + " or ".join(or_clauses) + ")"


def _build_azure_search_rule_filter(
    rule: Rule,
    user: dict[str, Any]
) -> Optional[str]:
    """Build an Azure AI Search filter for a single rule.

    Returns:
        - str: A filter expression for this rule
        - None: Rule matches all documents (no restrictions)
        - SKIP_RULE: User doesn't satisfy this rule's allow conditions
    """
    # Check if user satisfies the allow conditions
    if not user_satisfies_allow(rule.allow, user):
        return SKIP_RULE

    # User is allowed - build filter for document match conditions
    and_clauses = []

    # Add match conditions (what documents this rule applies to)
    if rule.match:
        for key, value in rule.match.items():
            if isinstance(value, list):
                # Match any value in list - use 'in' operator
                values_str = ", ".join(_azure_search_escape_value(v) for v in value)
                and_clauses.append(f"{key} in ({values_str})")
            else:
                # Single value - use 'eq'
                and_clauses.append(f"{key} eq {_azure_search_escape_value(value)}")

    # Add conditions that reference user context
    if rule.allow.conditions:
        for condition_str in rule.allow.conditions:
            # Compile the condition and build filter
            from ...policy.compiler import ConditionCompiler

            try:
                compiled = ConditionCompiler.compile_expression(condition_str)
                condition_filter = _build_azure_search_from_compiled_node(compiled, user)
                if condition_filter:
                    and_clauses.append(condition_filter)
            except (ValueError, AttributeError, KeyError, TypeError):
                # If condition can't be converted to native filter, skip it
                # Only catch specific exceptions, not system exceptions
                pass

    if not and_clauses:
        # No match conditions - this rule applies to all documents for this user
        # Return None to indicate no filter needed (matches all)
        return None

    if len(and_clauses) == 1:
        return and_clauses[0]

    # Multiple conditions: AND them together
    return "(" + " and ".join(and_clauses) + ")"


def _build_azure_search_from_compiled_node(
    node: Union['CompiledCondition', 'CompiledExpression'],
    user: dict[str, Any]
) -> Optional[str]:
    """
    Build an Azure AI Search filter from a compiled expression node.

    This enables native OR/AND support by converting the expression tree
    to OData filter syntax.

    Args:
        node: CompiledCondition or CompiledExpression from compiler
        user: User context

    Returns:
        OData filter string with native OR/AND support
    """
    from ...policy.compiler import CompiledCondition, CompiledExpression, LogicalOperator

    if isinstance(node, CompiledCondition):
        # Convert single condition to Azure filter
        return _build_azure_search_from_condition(node, user)

    elif isinstance(node, CompiledExpression):
        # Convert OR/AND expression tree
        child_filters = []

        for child in node.children:
            child_filter = _build_azure_search_from_compiled_node(child, user)
            if child_filter is not None:
                child_filters.append(child_filter)

        if not child_filters:
            return None

        if len(child_filters) == 1:
            return child_filters[0]

        # Build native OR/AND filter
        if node.operator == LogicalOperator.OR:
            # OR: Use 'or' operator
            return "(" + " or ".join(child_filters) + ")"
        elif node.operator == LogicalOperator.AND:
            # AND: Use 'and' operator
            return "(" + " and ".join(child_filters) + ")"

    return None


def _build_azure_search_from_condition(
    condition: 'CompiledCondition',
    user: dict[str, Any]
) -> Optional[str]:
    """
    Build an Azure AI Search filter from a CompiledCondition.

    Converts the compiled AST node to native OData filter syntax.
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

    # Convert condition to Azure filter based on operator
    if condition.operator == ConditionOperator.EQUALS:
        # user.field == document.field OR document.field == user.field
        if condition.left.value_type == ValueType.USER_FIELD and \
           condition.right and condition.right.value_type == ValueType.DOCUMENT_FIELD:
            # user.field == document.field
            doc_field = _validate_azure_field_path(condition.right.field_path)
            return f"{doc_field} eq {_azure_search_escape_value(left_value)}"
        elif condition.left.value_type == ValueType.DOCUMENT_FIELD and \
             condition.right and condition.right.value_type == ValueType.USER_FIELD:
            # document.field == user.field
            doc_field = _validate_azure_field_path(condition.left.field_path)
            return f"{doc_field} eq {_azure_search_escape_value(right_value)}"
        elif condition.left.value_type == ValueType.DOCUMENT_FIELD and \
             condition.right and condition.right.value_type in (
                 ValueType.LITERAL_STRING,
                 ValueType.LITERAL_NUMBER,
                 ValueType.LITERAL_BOOL
             ):
            # document.field == literal
            doc_field = _validate_azure_field_path(condition.left.field_path)
            return f"{doc_field} eq {_azure_search_escape_value(right_value)}"

    elif condition.operator == ConditionOperator.NOT_EQUALS:
        # user.field != document.field
        if condition.left.value_type == ValueType.USER_FIELD and \
           condition.right and condition.right.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = _validate_azure_field_path(condition.right.field_path)
            return f"{doc_field} ne {_azure_search_escape_value(left_value)}"
        elif (condition.left.value_type == ValueType.DOCUMENT_FIELD and \
             condition.right and condition.right.value_type == ValueType.USER_FIELD) or (condition.left.value_type == ValueType.DOCUMENT_FIELD and \
             condition.right and condition.right.value_type in (
                 ValueType.LITERAL_STRING,
                 ValueType.LITERAL_NUMBER,
                 ValueType.LITERAL_BOOL
             )):
            doc_field = _validate_azure_field_path(condition.left.field_path)
            return f"{doc_field} ne {_azure_search_escape_value(right_value)}"

    elif condition.operator == ConditionOperator.IN:
        # Check condition type
        if condition.condition_type == ConditionType.VALUE_IN_ARRAY_FIELD:
            # user.id in document.allowed_users
            # Azure AI Search uses any() for collection membership
            if condition.left.value_type == ValueType.USER_FIELD and \
               condition.right and condition.right.value_type == ValueType.DOCUMENT_FIELD:
                doc_field = _validate_azure_field_path(condition.right.field_path)
                # Use collection/any lambda expression
                return f"{doc_field}/any(x: x eq {_azure_search_escape_value(left_value)})"
        elif condition.left.value_type == ValueType.DOCUMENT_FIELD and \
             condition.right and condition.right.value_type == ValueType.LITERAL_LIST:
            # document.field in ['a', 'b', 'c']
            doc_field = _validate_azure_field_path(condition.left.field_path)
            values_str = ", ".join(_azure_search_escape_value(v) for v in right_value)
            return f"{doc_field} in ({values_str})"

    elif condition.operator == ConditionOperator.NOT_IN:
        if condition.condition_type == ConditionType.VALUE_IN_ARRAY_FIELD:
            # user.id not in document.allowed_users
            # Azure AI Search uses all() with ne for "not in collection"
            if condition.left.value_type == ValueType.USER_FIELD and \
               condition.right and condition.right.value_type == ValueType.DOCUMENT_FIELD:
                doc_field = _validate_azure_field_path(condition.right.field_path)
                # Use collection/all lambda expression with ne
                return f"{doc_field}/all(x: x ne {_azure_search_escape_value(left_value)})"
        elif condition.left.value_type == ValueType.DOCUMENT_FIELD and \
             condition.right and condition.right.value_type == ValueType.LITERAL_LIST:
            # document.field not in ['a', 'b', 'c']
            # OData doesn't have "not in", use multiple ne with and
            doc_field = _validate_azure_field_path(condition.left.field_path)
            ne_clauses = [f"{doc_field} ne {_azure_search_escape_value(v)}" for v in right_value]
            return "(" + " and ".join(ne_clauses) + ")"

    elif condition.operator == ConditionOperator.EXISTS:
        # Check if field exists (not null)
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = _validate_azure_field_path(condition.left.field_path)
            return f"{doc_field} ne null"

    elif condition.operator == ConditionOperator.NOT_EXISTS:
        # Check if field doesn't exist (is null)
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = _validate_azure_field_path(condition.left.field_path)
            return f"{doc_field} eq null"

    elif condition.operator == ConditionOperator.GREATER_THAN:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = _validate_azure_field_path(condition.left.field_path)
            value = right_value if condition.right else None
            if value is not None:
                return f"{doc_field} gt {_azure_search_escape_value(value)}"

    elif condition.operator == ConditionOperator.GREATER_THAN_OR_EQUAL:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = _validate_azure_field_path(condition.left.field_path)
            value = right_value if condition.right else None
            if value is not None:
                return f"{doc_field} ge {_azure_search_escape_value(value)}"

    elif condition.operator == ConditionOperator.LESS_THAN:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = _validate_azure_field_path(condition.left.field_path)
            value = right_value if condition.right else None
            if value is not None:
                return f"{doc_field} lt {_azure_search_escape_value(value)}"

    elif condition.operator == ConditionOperator.LESS_THAN_OR_EQUAL:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = _validate_azure_field_path(condition.left.field_path)
            value = right_value if condition.right else None
            if value is not None:
                return f"{doc_field} le {_azure_search_escape_value(value)}"

    # If we reach here with an unhandled operator, raise an error instead of silently allowing
    raise UnsupportedConditionError(
        condition=str(condition.operator),
        backend="azure_search",
        reason=f"Operator {condition.operator} with value types "
               f"left={condition.left.value_type}, right={condition.right.value_type if condition.right else None} "
               f"is not supported. This is a security measure to prevent silent filter bypass."
    )


def _azure_search_escape_value(value: Any) -> str:
    """
    Escape a value for Azure AI Search OData filter.

    Args:
        value: Value to escape

    Returns:
        Escaped string representation
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        # OData strings use single quotes, escape single quotes by doubling
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    elif value is None:
        return "null"
    else:
        # Fallback for other types
        return str(value)
