"""
Abstract base class for filter builders.

Provides common policy traversal logic to reduce code duplication across
backend-specific filter builders. Each backend only needs to implement
the output format-specific methods.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Optional, TypeVar, Union

from ..exceptions import UnsupportedConditionError
from .base import (
    DENY_ALL_FIELD,
    DENY_ALL_VALUE,
    SKIP_RULE,
    get_nested_value,
    parse_list_literal,
    parse_literal_value,
    user_satisfies_allow,
    validate_field_path,
)

if TYPE_CHECKING:
    from ..policy.compiler import (
        CompiledCondition,
        CompiledExpression,
    )
    from ..policy.models import Policy, Rule

# Type variable for the filter output type (dict, str, object, etc.)
T = TypeVar("T")


class FilterBuilderBase(ABC, Generic[T]):
    """
    Abstract base class for filter builders.

    Subclasses must implement the abstract methods to define how filters
    are represented for their specific backend.

    Type parameter T represents the filter output type:
    - dict for Pinecone, ChromaDB, Milvus
    - str for SQL-based backends (pgvector, Elasticsearch)
    - object for Qdrant (models.Filter)
    """

    # Subclass should set this
    backend_name: str = "generic"

    # Field path separator (. for most, / for Azure)
    field_separator: str = "."

    def build_filter(
        self,
        policy: "Policy",
        user: dict[str, Any]
    ) -> Optional[T]:
        """
        Build a filter from a policy and user context.

        This is the main entry point that handles the common policy
        traversal logic across all backends.

        Args:
            policy: The access control policy
            user: User context dictionary

        Returns:
            Backend-specific filter, or None if no filter needed
        """
        or_clauses = []

        for rule in policy.rules:
            rule_filter = self._build_rule_filter(rule, user)
            if rule_filter is SKIP_RULE:
                continue
            if rule_filter is None:
                # Rule grants unrestricted access
                return None
            or_clauses.append(rule_filter)

        if not or_clauses:
            if policy.default == "allow":
                return None
            else:
                return self._deny_all()

        if len(or_clauses) == 1:
            return or_clauses[0]

        return self._combine_or(or_clauses)

    def _build_rule_filter(
        self,
        rule: "Rule",
        user: dict[str, Any]
    ) -> Optional[T]:
        """Build filter for a single rule."""
        if not user_satisfies_allow(rule.allow, user):
            return SKIP_RULE

        and_clauses = []

        # Process match conditions
        if rule.match:
            for field, value in rule.match.items():
                match_filter = self._build_match_filter(field, value)
                if match_filter is not None:
                    and_clauses.append(match_filter)

        # Process dynamic conditions
        if rule.allow.conditions:
            for condition_str in rule.allow.conditions:
                cond_filter = self._build_condition_filter(condition_str, user)
                if cond_filter is not None:
                    and_clauses.append(cond_filter)

        if not and_clauses:
            return None  # No restrictions

        if len(and_clauses) == 1:
            return and_clauses[0]

        return self._combine_and(and_clauses)

    def _build_condition_filter(
        self,
        condition: str,
        user: dict[str, Any]
    ) -> Optional[T]:
        """
        Build filter from a condition string.

        Tries compiled expression first, falls back to legacy parsing.
        """
        import re

        condition = condition.strip()

        # Check for OR/AND expressions
        if re.search(r'\b(OR|AND)\b', condition, re.IGNORECASE):
            from ..policy.compiler import CompiledExpression, ConditionCompiler

            try:
                compiled = ConditionCompiler.compile_expression(condition)
                if isinstance(compiled, CompiledExpression):
                    return self._build_from_compiled_node(compiled, user)
            except (ValueError, AttributeError, KeyError, TypeError):
                pass

        # Try compiled single condition
        from ..policy.compiler import ConditionCompiler

        try:
            compiled = ConditionCompiler.compile_expression(condition)
            return self._build_from_compiled_node(compiled, user)
        except (ValueError, AttributeError, KeyError, TypeError):
            pass

        # Fall back to legacy string parsing
        return self._parse_condition_string(condition, user)

    def _build_from_compiled_node(
        self,
        node: Union["CompiledCondition", "CompiledExpression"],
        user: dict[str, Any]
    ) -> Optional[T]:
        """Build filter from a compiled expression tree."""
        from ..policy.compiler import CompiledCondition, CompiledExpression, LogicalOperator

        if isinstance(node, CompiledCondition):
            return self._build_from_condition(node, user)

        elif isinstance(node, CompiledExpression):
            child_filters = []

            for child in node.children:
                child_filter = self._build_from_compiled_node(child, user)
                if child_filter is not None:
                    child_filters.append(child_filter)

            if not child_filters:
                return None

            if len(child_filters) == 1:
                return child_filters[0]

            if node.operator == LogicalOperator.OR:
                return self._combine_or(child_filters)
            else:
                return self._combine_and(child_filters)

        return None

    def _build_from_condition(
        self,
        condition: "CompiledCondition",
        user: dict[str, Any]
    ) -> Optional[T]:
        """
        Build filter from a CompiledCondition.

        This method handles the common value resolution logic and
        delegates format-specific output to abstract methods.
        """
        from ..policy.compiler import (
            CompiledConditionEvaluator,
            ValueType,
        )

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
                ValueType.LITERAL_LIST,
            ):
                right_value = condition.right.value
            elif condition.right.value_type == ValueType.LITERAL_NONE:
                right_value = None

        # Get document field if present
        doc_field = None
        if condition.left.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = self._format_field_path(condition.left.field_path)
        elif condition.right and condition.right.value_type == ValueType.DOCUMENT_FIELD:
            doc_field = self._format_field_path(condition.right.field_path)

        # Dispatch to operator-specific handler
        return self._handle_operator(
            condition=condition,
            left_value=left_value,
            right_value=right_value,
            doc_field=doc_field,
            user=user,
        )

    def _handle_operator(
        self,
        condition: "CompiledCondition",
        left_value: Any,
        right_value: Any,
        doc_field: Optional[str],
        user: dict[str, Any],
    ) -> Optional[T]:
        """
        Handle operator dispatch with common patterns.

        Subclasses can override specific operator handling.
        """
        from ..policy.compiler import ConditionOperator, ValueType

        op = condition.operator
        left_type = condition.left.value_type
        right_type = condition.right.value_type if condition.right else None

        # EXISTS
        if op == ConditionOperator.EXISTS:
            if left_type == ValueType.DOCUMENT_FIELD:
                return self._filter_exists(doc_field)

        # NOT_EXISTS
        elif op == ConditionOperator.NOT_EXISTS:
            if left_type == ValueType.DOCUMENT_FIELD:
                return self._filter_not_exists(doc_field)

        # EQUALS
        elif op == ConditionOperator.EQUALS:
            # user.field == document.field
            if left_type == ValueType.USER_FIELD and right_type == ValueType.DOCUMENT_FIELD:
                if left_value is None:
                    return self._deny_all()
                return self._filter_equals(doc_field, left_value)

            # document.field == user.field
            elif left_type == ValueType.DOCUMENT_FIELD and right_type == ValueType.USER_FIELD:
                if right_value is None:
                    return self._deny_all()
                return self._filter_equals(doc_field, right_value)

            # document.field == literal
            elif left_type == ValueType.DOCUMENT_FIELD and right_type in (
                ValueType.LITERAL_STRING,
                ValueType.LITERAL_NUMBER,
                ValueType.LITERAL_BOOL,
                ValueType.LITERAL_NONE,
            ):
                return self._filter_equals(doc_field, right_value)

            # user.field == literal (evaluate at build time)
            elif left_type == ValueType.USER_FIELD and right_type in (
                ValueType.LITERAL_STRING,
                ValueType.LITERAL_NUMBER,
                ValueType.LITERAL_BOOL,
            ):
                if left_value == right_value:
                    return None  # Condition true, no filter needed
                else:
                    return self._deny_all()

        # NOT_EQUALS
        elif op == ConditionOperator.NOT_EQUALS:
            if left_type == ValueType.DOCUMENT_FIELD:
                return self._filter_not_equals(doc_field, right_value)

        # IN
        elif op == ConditionOperator.IN:
            # user.id in document.array_field
            if left_type == ValueType.USER_FIELD and right_type == ValueType.DOCUMENT_FIELD:
                if left_value is None:
                    return self._deny_all()
                return self._filter_value_in_array(doc_field, left_value)

            # document.field in [literals]
            elif left_type == ValueType.DOCUMENT_FIELD and isinstance(right_value, list):
                if len(right_value) == 0:
                    return self._deny_all()
                return self._filter_field_in_list(doc_field, right_value)

        # NOT_IN
        elif op == ConditionOperator.NOT_IN:
            # user.id not in document.array_field
            if left_type == ValueType.USER_FIELD and right_type == ValueType.DOCUMENT_FIELD:
                if left_value is None:
                    return None  # User not in anything
                return self._filter_value_not_in_array(doc_field, left_value)

            # document.field not in [literals]
            elif left_type == ValueType.DOCUMENT_FIELD and isinstance(right_value, list):
                if len(right_value) == 0:
                    return None  # Nothing excluded
                return self._filter_field_not_in_list(doc_field, right_value)

        # Comparison operators
        elif op == ConditionOperator.LESS_THAN:
            if left_type == ValueType.DOCUMENT_FIELD:
                return self._filter_less_than(doc_field, right_value)
            elif left_type == ValueType.USER_FIELD and right_type == ValueType.LITERAL_NUMBER:
                if left_value is not None and left_value < right_value:
                    return None
                return self._deny_all()

        elif op == ConditionOperator.LESS_THAN_OR_EQUAL:
            if left_type == ValueType.DOCUMENT_FIELD:
                return self._filter_less_than_or_equal(doc_field, right_value)
            elif left_type == ValueType.USER_FIELD and right_type == ValueType.LITERAL_NUMBER:
                if left_value is not None and left_value <= right_value:
                    return None
                return self._deny_all()

        elif op == ConditionOperator.GREATER_THAN:
            if left_type == ValueType.DOCUMENT_FIELD:
                return self._filter_greater_than(doc_field, right_value)
            elif left_type == ValueType.USER_FIELD and right_type == ValueType.LITERAL_NUMBER:
                if left_value is not None and left_value > right_value:
                    return None
                return self._deny_all()

        elif op == ConditionOperator.GREATER_THAN_OR_EQUAL:
            if left_type == ValueType.DOCUMENT_FIELD:
                return self._filter_greater_than_or_equal(doc_field, right_value)
            elif left_type == ValueType.USER_FIELD and right_type == ValueType.LITERAL_NUMBER:
                if left_value is not None and left_value >= right_value:
                    return None
                return self._deny_all()

        # Unhandled operator - raise error
        raise UnsupportedConditionError(
            condition=str(op),
            backend=self.backend_name,
            reason=f"Operator {op} with value types "
                   f"left={left_type}, right={right_type} "
                   f"is not supported. This is a security measure to prevent silent filter bypass."
        )

    def _format_field_path(self, field_path: list[str]) -> str:
        """Format field path for this backend."""
        # Validate first
        validate_field_path(field_path, self.backend_name)
        return self.field_separator.join(field_path)

    # =========================================================================
    # Abstract methods - must be implemented by each backend
    # =========================================================================

    @abstractmethod
    def _deny_all(self) -> T:
        """Return a filter that matches nothing (deny all access)."""
        pass

    @abstractmethod
    def _combine_or(self, filters: list[T]) -> T:
        """Combine multiple filters with OR logic."""
        pass

    @abstractmethod
    def _combine_and(self, filters: list[T]) -> T:
        """Combine multiple filters with AND logic."""
        pass

    @abstractmethod
    def _build_match_filter(self, field: str, value: Any) -> Optional[T]:
        """Build filter for a match condition (field == value or field in values)."""
        pass

    @abstractmethod
    def _filter_equals(self, field: str, value: Any) -> T:
        """Build equality filter: field == value."""
        pass

    @abstractmethod
    def _filter_not_equals(self, field: str, value: Any) -> T:
        """Build inequality filter: field != value."""
        pass

    @abstractmethod
    def _filter_exists(self, field: str) -> T:
        """Build exists filter: field exists (is not null)."""
        pass

    @abstractmethod
    def _filter_not_exists(self, field: str) -> T:
        """Build not exists filter: field does not exist (is null)."""
        pass

    @abstractmethod
    def _filter_value_in_array(self, field: str, value: Any) -> T:
        """Build filter: value in field (field is array containing value)."""
        pass

    @abstractmethod
    def _filter_value_not_in_array(self, field: str, value: Any) -> T:
        """Build filter: value not in field (field does not contain value)."""
        pass

    @abstractmethod
    def _filter_field_in_list(self, field: str, values: list) -> T:
        """Build filter: field in [values] (field equals one of the values)."""
        pass

    @abstractmethod
    def _filter_field_not_in_list(self, field: str, values: list) -> T:
        """Build filter: field not in [values]."""
        pass

    @abstractmethod
    def _filter_less_than(self, field: str, value: Any) -> T:
        """Build filter: field < value."""
        pass

    @abstractmethod
    def _filter_less_than_or_equal(self, field: str, value: Any) -> T:
        """Build filter: field <= value."""
        pass

    @abstractmethod
    def _filter_greater_than(self, field: str, value: Any) -> T:
        """Build filter: field > value."""
        pass

    @abstractmethod
    def _filter_greater_than_or_equal(self, field: str, value: Any) -> T:
        """Build filter: field >= value."""
        pass

    def _parse_condition_string(
        self,
        condition: str,
        user: dict[str, Any]
    ) -> Optional[T]:
        """
        Parse legacy condition string format.

        Default implementation for common patterns. Subclasses can override
        for backend-specific parsing.
        """
        condition = condition.strip()

        # EXISTS
        if " not exists" in condition:
            field = condition.replace(" not exists", "").strip()
            if field.startswith("document."):
                return self._filter_not_exists(field[9:])
            return None

        elif " exists" in condition:
            field = condition.replace(" exists", "").strip()
            if field.startswith("document."):
                return self._filter_exists(field[9:])
            return None

        # NOT_EQUALS
        elif "!=" in condition:
            parts = condition.split("!=", 1)
            if len(parts) != 2:
                return None

            left, right = parts[0].strip(), parts[1].strip()

            if left.startswith("document."):
                doc_field = left[9:]
                value = parse_literal_value(right)
                return self._filter_not_equals(doc_field, value)

        # EQUALS
        elif "==" in condition:
            parts = condition.split("==", 1)
            if len(parts) != 2:
                return None

            left, right = parts[0].strip(), parts[1].strip()

            # user.field == document.field
            if left.startswith("user.") and right.startswith("document."):
                user_field = left[5:]
                doc_field = right[9:]
                user_value = get_nested_value(user, user_field)

                if user_value is None:
                    return self._deny_all()
                return self._filter_equals(doc_field, user_value)

            # document.field == literal
            elif left.startswith("document."):
                doc_field = left[9:]
                value = parse_literal_value(right)
                return self._filter_equals(doc_field, value)

        # NOT IN
        elif " not in " in condition:
            parts = condition.split(" not in ", 1)
            if len(parts) != 2:
                return None

            left, right = parts[0].strip(), parts[1].strip()

            # user.field not in document.array
            if left.startswith("user.") and right.startswith("document."):
                user_field = left[5:]
                doc_field = right[9:]
                user_value = get_nested_value(user, user_field)

                if user_value is None:
                    return None
                return self._filter_value_not_in_array(doc_field, user_value)

            # document.field not in [list]
            elif left.startswith("document."):
                doc_field = left[9:]
                list_values = parse_list_literal(right)

                if list_values is not None:
                    if len(list_values) == 0:
                        return None
                    return self._filter_field_not_in_list(doc_field, list_values)

        # IN
        elif " in " in condition:
            parts = condition.split(" in ", 1)
            if len(parts) != 2:
                return None

            left, right = parts[0].strip(), parts[1].strip()

            # user.field in document.array
            if left.startswith("user.") and right.startswith("document."):
                user_field = left[5:]
                doc_field = right[9:]
                user_value = get_nested_value(user, user_field)

                if user_value is None:
                    return self._deny_all()
                return self._filter_value_in_array(doc_field, user_value)

            # document.field in [list]
            elif left.startswith("document."):
                doc_field = left[9:]
                list_values = parse_list_literal(right)

                if list_values is not None:
                    if len(list_values) == 0:
                        return self._deny_all()
                    return self._filter_field_in_list(doc_field, list_values)

        return None


class DictFilterBuilder(FilterBuilderBase[dict]):
    """
    Base class for dict-based filter backends.

    Used by Pinecone, ChromaDB, and similar backends that use
    dict-based filters with $eq, $in, $and, $or operators.
    """

    # Operator names can be customized by subclasses
    op_eq: str = "$eq"
    op_ne: str = "$ne"
    op_in: str = "$in"
    op_nin: str = "$nin"
    op_lt: str = "$lt"
    op_lte: str = "$lte"
    op_gt: str = "$gt"
    op_gte: str = "$gte"
    op_exists: str = "$exists"  # Only used if supports_exists is True
    op_and: str = "$and"
    op_or: str = "$or"

    # Whether backend supports $exists operator
    supports_exists: bool = True

    # Whether backend uses $exists: True/False or $ne/$eq: None
    exists_uses_none: bool = False

    def _deny_all(self) -> dict:
        return {DENY_ALL_FIELD: {self.op_eq: DENY_ALL_VALUE}}

    def _combine_or(self, filters: list[dict]) -> dict:
        return {self.op_or: filters}

    def _combine_and(self, filters: list[dict]) -> dict:
        return {self.op_and: filters}

    def _build_match_filter(self, field: str, value: Any) -> Optional[dict]:
        if isinstance(value, list):
            return {field: {self.op_in: value}}
        return {field: {self.op_eq: value}}

    def _filter_equals(self, field: str, value: Any) -> dict:
        return {field: {self.op_eq: value}}

    def _filter_not_equals(self, field: str, value: Any) -> dict:
        return {field: {self.op_ne: value}}

    def _filter_exists(self, field: str) -> dict:
        if self.exists_uses_none:
            return {field: {self.op_ne: None}}
        return {field: {self.op_exists: True}}

    def _filter_not_exists(self, field: str) -> dict:
        if self.exists_uses_none:
            return {field: {self.op_eq: None}}
        return {field: {self.op_exists: False}}

    def _filter_value_in_array(self, field: str, value: Any) -> dict:
        return {field: {self.op_in: [value]}}

    def _filter_value_not_in_array(self, field: str, value: Any) -> dict:
        return {field: {self.op_nin: [value]}}

    def _filter_field_in_list(self, field: str, values: list) -> dict:
        return {field: {self.op_in: values}}

    def _filter_field_not_in_list(self, field: str, values: list) -> dict:
        return {field: {self.op_nin: values}}

    def _filter_less_than(self, field: str, value: Any) -> dict:
        return {field: {self.op_lt: value}}

    def _filter_less_than_or_equal(self, field: str, value: Any) -> dict:
        return {field: {self.op_lte: value}}

    def _filter_greater_than(self, field: str, value: Any) -> dict:
        return {field: {self.op_gt: value}}

    def _filter_greater_than_or_equal(self, field: str, value: Any) -> dict:
        return {field: {self.op_gte: value}}


# Concrete implementations for common backends
class PineconeFilterBuilder(DictFilterBuilder):
    """Filter builder for Pinecone."""

    backend_name = "pinecone"
    supports_exists = True
    exists_uses_none = False


class ChromaDBFilterBuilder(DictFilterBuilder):
    """Filter builder for ChromaDB."""

    backend_name = "chromadb"
    supports_exists = False  # ChromaDB uses $ne None
    exists_uses_none = True


# Export helper functions for backward compatibility
def to_pinecone_filter_v2(policy: "Policy", user: dict[str, Any]) -> Optional[dict]:
    """New implementation using FilterBuilderBase."""
    return PineconeFilterBuilder().build_filter(policy, user)


def to_chromadb_filter_v2(policy: "Policy", user: dict[str, Any]) -> Optional[dict]:
    """New implementation using FilterBuilderBase."""
    return ChromaDBFilterBuilder().build_filter(policy, user)
