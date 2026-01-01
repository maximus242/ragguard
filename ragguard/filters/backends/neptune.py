"""
Amazon Neptune filter builder.

Converts RAGGuard policies to Gremlin traversal predicates for Neptune.
"""

from typing import TYPE_CHECKING, Any, List, Optional, Union

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
    validate_field_name,
)

if TYPE_CHECKING:
    from ...policy.compiler import CompiledCondition, CompiledExpression


def to_neptune_filter(
    policy: Policy,
    user: dict[str, Any]
) -> List[dict]:
    """
    Convert a policy and user context to Gremlin filter predicates.

    Returns a list of filter specifications that can be applied to a
    Gremlin traversal using .has() steps.

    Args:
        policy: The access control policy
        user: User context

    Returns:
        List of filter dicts, each with:
        - "type": "has" | "hasNot" | "or"
        - "property": property name (for has/hasNot)
        - "predicate": predicate type ("eq", "neq", "within", "without", "gt", "lt", etc.)
        - "value": the value to compare against
        - "children": list of child filters (for "or" type)

    Example:
        >>> filters = to_neptune_filter(policy, user)
        >>> # Apply filters to traversal
        >>> g = graph.traversal()
        >>> t = g.V().hasLabel("Document")
        >>> for f in filters:
        ...     t = apply_filter(t, f)
    """
    or_filters = []

    for rule in policy.rules:
        rule_filters = _build_neptune_rule_filter(rule, user)
        if rule_filters is not None:
            or_filters.append(rule_filters)

    if not or_filters:
        # No rules apply to this user
        if policy.default == "allow":
            # Allow all - return empty filter list
            return []
        else:
            # Deny all - return filter that matches nothing
            return [{"type": "has", "property": DENY_ALL_FIELD, "predicate": "eq", "value": DENY_ALL_VALUE}]

    if len(or_filters) == 1:
        return or_filters[0]

    # Multiple rules: wrap in OR
    return [{"type": "or", "children": or_filters}]


def _build_neptune_rule_filter(
    rule: Rule,
    user: dict[str, Any]
) -> Optional[List[dict]]:
    """
    Build Gremlin filter predicates for a single rule.

    Returns a list of filter specs or None if rule doesn't apply.
    """
    # Check if user satisfies the allow conditions
    if not user_satisfies_allow(rule.allow, user):
        return None

    # User is allowed - build filters
    filters = []

    # Add match conditions
    if rule.match:
        for key, value in rule.match.items():
            safe_key = validate_field_name(key, "neptune")

            if isinstance(value, list):
                # Use "within" for list values
                filters.append({
                    "type": "has",
                    "property": safe_key,
                    "predicate": "within",
                    "value": value
                })
            else:
                filters.append({
                    "type": "has",
                    "property": safe_key,
                    "predicate": "eq",
                    "value": value
                })

    # Add conditions that reference user context
    if rule.allow.conditions:
        for condition in rule.allow.conditions:
            cond_filters = _build_neptune_condition_filter(condition, user)
            if cond_filters:
                filters.extend(cond_filters)

    if not filters:
        # No conditions - allow all for this user
        return []

    return filters


def _build_neptune_from_compiled_node(
    node: Union['CompiledCondition', 'CompiledExpression'],
    user: dict[str, Any]
) -> Optional[List[dict]]:
    """
    Build Gremlin filter predicates from a compiled expression node.
    """
    from ...policy.compiler import CompiledCondition, CompiledExpression, LogicalOperator

    if isinstance(node, CompiledCondition):
        return _build_neptune_from_condition(node, user)

    elif isinstance(node, CompiledExpression):
        child_filters = []

        for child in node.children:
            child_filter = _build_neptune_from_compiled_node(child, user)
            if child_filter:
                child_filters.append(child_filter)

        if not child_filters:
            return None

        if len(child_filters) == 1:
            return child_filters[0]

        # Build OR/AND structure
        if node.operator == LogicalOperator.OR:
            return [{"type": "or", "children": child_filters}]
        else:  # AND - flatten into list
            result = []
            for cf in child_filters:
                result.extend(cf)
            return result

    return None


def _build_neptune_from_condition(
    condition: 'CompiledCondition',
    user: dict[str, Any]
) -> Optional[List[dict]]:
    """
    Build Gremlin filter predicates from a CompiledCondition.
    """
    from ...policy.compiler import CompiledConditionEvaluator, ConditionOperator, ValueType

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

    # Get document field
    doc_field = None
    if condition.left.value_type == ValueType.DOCUMENT_FIELD:
        doc_field = ".".join(condition.left.field_path)
        doc_field = validate_field_name(doc_field, "neptune")
    elif condition.right and condition.right.value_type == ValueType.DOCUMENT_FIELD:
        doc_field = ".".join(condition.right.field_path)
        doc_field = validate_field_name(doc_field, "neptune")

    # Handle EXISTS
    if condition.operator == ConditionOperator.EXISTS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            return [{"type": "has", "property": doc_field, "predicate": "exists", "value": True}]

    # Handle NOT_EXISTS
    elif condition.operator == ConditionOperator.NOT_EXISTS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            return [{"type": "hasNot", "property": doc_field}]

    # Handle EQUALS
    elif condition.operator == ConditionOperator.EQUALS:
        if (condition.left.value_type == ValueType.USER_FIELD and
                condition.right.value_type == ValueType.DOCUMENT_FIELD):
            if left_value is None:
                return [{"type": "has", "property": DENY_ALL_FIELD, "predicate": "eq", "value": DENY_ALL_VALUE}]
            return [{"type": "has", "property": doc_field, "predicate": "eq", "value": left_value}]

        elif condition.left.value_type == ValueType.DOCUMENT_FIELD:
            return [{"type": "has", "property": doc_field, "predicate": "eq", "value": right_value}]

    # Handle NOT_EQUALS
    elif condition.operator == ConditionOperator.NOT_EQUALS:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            return [{"type": "has", "property": doc_field, "predicate": "neq", "value": right_value}]

    # Handle IN
    elif condition.operator == ConditionOperator.IN:
        if (condition.left.value_type == ValueType.USER_FIELD and
                condition.right.value_type == ValueType.DOCUMENT_FIELD):
            if left_value is None:
                return [{"type": "has", "property": DENY_ALL_FIELD, "predicate": "eq", "value": DENY_ALL_VALUE}]
            # Check if user value is in document's list property
            return [{"type": "has", "property": doc_field, "predicate": "containing", "value": left_value}]

        elif (condition.left.value_type == ValueType.DOCUMENT_FIELD and
              right_value is not None and isinstance(right_value, list)):
            return [{"type": "has", "property": doc_field, "predicate": "within", "value": right_value}]

    # Handle NOT_IN
    elif condition.operator == ConditionOperator.NOT_IN:
        if (condition.left.value_type == ValueType.USER_FIELD and
                condition.right.value_type == ValueType.DOCUMENT_FIELD):
            if left_value is None:
                return []
            return [{"type": "has", "property": doc_field, "predicate": "notContaining", "value": left_value}]

        elif (condition.left.value_type == ValueType.DOCUMENT_FIELD and
              right_value is not None and isinstance(right_value, list)):
            return [{"type": "has", "property": doc_field, "predicate": "without", "value": right_value}]

    # Handle comparison operators
    elif condition.operator == ConditionOperator.GREATER_THAN:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            return [{"type": "has", "property": doc_field, "predicate": "gt", "value": right_value}]

    elif condition.operator == ConditionOperator.LESS_THAN:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            return [{"type": "has", "property": doc_field, "predicate": "lt", "value": right_value}]

    elif condition.operator == ConditionOperator.GREATER_THAN_OR_EQUAL:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            return [{"type": "has", "property": doc_field, "predicate": "gte", "value": right_value}]

    elif condition.operator == ConditionOperator.LESS_THAN_OR_EQUAL:
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            return [{"type": "has", "property": doc_field, "predicate": "lte", "value": right_value}]

    # If we reach here with an unhandled operator, raise an error instead of silently allowing
    raise UnsupportedConditionError(
        condition=str(condition.operator),
        backend="neptune",
        reason=f"Operator {condition.operator} with value types "
               f"left={condition.left.value_type}, right={condition.right.value_type if condition.right else None} "
               f"is not supported. This is a security measure to prevent silent filter bypass."
    )


def _build_neptune_condition_filter(
    condition: str,
    user: dict[str, Any]
) -> Optional[List[dict]]:
    """
    Build Gremlin filter predicates from a condition expression.
    """
    from ...policy.compiler import CompiledExpression, ConditionCompiler

    # Try to compile as expression
    try:
        compiled = ConditionCompiler.compile_expression(condition)

        if isinstance(compiled, CompiledExpression):
            return _build_neptune_from_compiled_node(compiled, user)
        else:
            return _build_neptune_from_condition(compiled, user)
    except (ValueError, AttributeError, KeyError, TypeError) as e:
        logger.debug(
            "Condition compilation failed, falling back to string parsing: %s (condition: %s)",
            str(e), condition
        )

    # Fallback to string parsing
    return _parse_condition_string(condition, user)


def _parse_condition_string(
    condition: str,
    user: dict[str, Any]
) -> Optional[List[dict]]:
    """
    Parse a condition string and convert to Gremlin filter specs.
    """
    condition = condition.strip()

    # Parse field existence
    if " not exists" in condition:
        field = condition.replace(" not exists", "").strip()
        if field.startswith("document."):
            doc_field = validate_field_name(field[9:], "neptune")
            return [{"type": "hasNot", "property": doc_field}]
        return None

    elif " exists" in condition:
        field = condition.replace(" exists", "").strip()
        if field.startswith("document."):
            doc_field = validate_field_name(field[9:], "neptune")
            return [{"type": "has", "property": doc_field, "predicate": "exists", "value": True}]
        return None

    # Parse !=
    elif "!=" in condition:
        parts = condition.split("!=", 1)
        if len(parts) != 2:
            return None

        left, right = parts[0].strip(), parts[1].strip()

        if left.startswith("document."):
            doc_field = validate_field_name(left[9:], "neptune")
            literal_value = parse_literal_value(right)
            return [{"type": "has", "property": doc_field, "predicate": "neq", "value": literal_value}]

    # Parse ==
    elif "==" in condition:
        parts = condition.split("==", 1)
        if len(parts) != 2:
            return None

        left, right = parts[0].strip(), parts[1].strip()

        if left.startswith("user."):
            user_field = left[5:]
            user_value = get_nested_value(user, user_field)

            if user_value is None:
                return [{"type": "has", "property": DENY_ALL_FIELD, "predicate": "eq", "value": DENY_ALL_VALUE}]

            if right.startswith("document."):
                doc_field = validate_field_name(right[9:], "neptune")
                return [{"type": "has", "property": doc_field, "predicate": "eq", "value": user_value}]

        elif left.startswith("document."):
            doc_field = validate_field_name(left[9:], "neptune")
            literal_value = parse_literal_value(right)
            return [{"type": "has", "property": doc_field, "predicate": "eq", "value": literal_value}]

    # Parse "not in"
    elif " not in " in condition:
        parts = condition.split(" not in ", 1)
        if len(parts) != 2:
            return None

        left, right = parts[0].strip(), parts[1].strip()

        if left.startswith("user.") and right.startswith("document."):
            user_field = left[5:]
            doc_field = validate_field_name(right[9:], "neptune")
            user_value = get_nested_value(user, user_field)

            if user_value is None:
                return []

            return [{"type": "has", "property": doc_field, "predicate": "notContaining", "value": user_value}]

        elif left.startswith("document."):
            doc_field = validate_field_name(left[9:], "neptune")
            list_values = parse_list_literal(right)

            if list_values is not None and len(list_values) > 0:
                return [{"type": "has", "property": doc_field, "predicate": "without", "value": list_values}]

    # Parse "in"
    elif " in " in condition:
        parts = condition.split(" in ", 1)
        if len(parts) != 2:
            return None

        left, right = parts[0].strip(), parts[1].strip()

        if left.startswith("user.") and right.startswith("document."):
            user_field = left[5:]
            doc_field = validate_field_name(right[9:], "neptune")
            user_value = get_nested_value(user, user_field)

            if user_value is None:
                return [{"type": "has", "property": DENY_ALL_FIELD, "predicate": "eq", "value": DENY_ALL_VALUE}]

            return [{"type": "has", "property": doc_field, "predicate": "containing", "value": user_value}]

        elif left.startswith("document."):
            doc_field = validate_field_name(left[9:], "neptune")
            list_values = parse_list_literal(right)

            if list_values is not None:
                if len(list_values) == 0:
                    return [{"type": "has", "property": DENY_ALL_FIELD, "predicate": "eq", "value": DENY_ALL_VALUE}]
                return [{"type": "has", "property": doc_field, "predicate": "within", "value": list_values}]

    return None


def apply_neptune_filters(traversal: Any, filters: List[dict]) -> Any:
    """
    Apply filter specifications to a Gremlin traversal.

    This is a helper function to apply the filter specs returned by
    to_neptune_filter() to an actual Gremlin traversal object.

    Args:
        traversal: A Gremlin traversal (from gremlin_python)
        filters: List of filter specifications

    Returns:
        Modified traversal with filters applied

    Example:
        >>> from gremlin_python.process.graph_traversal import __
        >>> filters = to_neptune_filter(policy, user)
        >>> g = graph.traversal()
        >>> t = g.V().hasLabel("Document")
        >>> t = apply_neptune_filters(t, filters)
    """
    try:
        from gremlin_python.process.graph_traversal import __
        from gremlin_python.process.traversal import P
    except ImportError:
        raise FilterBuildError(
            "gremlin_python not installed. Install with: pip install gremlinpython"
        )

    def apply_single_filter(t, f):
        """Apply a single filter to the traversal."""
        if f["type"] == "or":
            # Build OR traversal
            or_traversals = []
            for child_filters in f["children"]:
                # Each child is a list of AND filters
                child_t = __.identity()
                for cf in child_filters:
                    child_t = apply_single_filter(child_t, cf)
                or_traversals.append(child_t)

            if len(or_traversals) == 1:
                return t.where(or_traversals[0])
            return t.or_(*or_traversals)

        elif f["type"] == "hasNot":
            return t.hasNot(f["property"])

        elif f["type"] == "has":
            prop = f["property"]
            pred = f["predicate"]
            value = f["value"]

            if pred == "eq":
                return t.has(prop, value)
            elif pred == "neq":
                return t.has(prop, P.neq(value))
            elif pred == "within":
                return t.has(prop, P.within(*value))
            elif pred == "without":
                return t.has(prop, P.without(*value))
            elif pred == "gt":
                return t.has(prop, P.gt(value))
            elif pred == "gte":
                return t.has(prop, P.gte(value))
            elif pred == "lt":
                return t.has(prop, P.lt(value))
            elif pred == "lte":
                return t.has(prop, P.lte(value))
            elif pred == "containing":
                # For checking if value is in a list property
                return t.has(prop, P.containing(value))
            elif pred == "notContaining":
                return t.has(prop, P.not_(P.containing(value)))
            elif pred == "exists":
                return t.has(prop)
            else:
                logger.warning(f"Unknown Gremlin predicate: {pred}")
                return t

        return t

    for f in filters:
        traversal = apply_single_filter(traversal, f)

    return traversal
