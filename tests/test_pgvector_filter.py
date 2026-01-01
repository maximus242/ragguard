"""
Comprehensive tests for pgvector filter builder to maximize coverage.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestPgvectorCompiledExpressionNode:
    """Tests for _build_pgvector_from_compiled_node with expressions."""

    def test_compiled_expression_empty_children(self):
        """Test compiled expression with no valid children."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_compiled_node
        from ragguard.policy.compiler import CompiledExpression, LogicalOperator

        # Create expression with empty children
        expr = CompiledExpression(
            operator=LogicalOperator.OR,
            children=[],
            original="empty"
        )

        result, params = _build_pgvector_from_compiled_node(expr, {"id": "alice"})
        assert result == ""
        assert params == []

    def test_compiled_expression_single_child(self):
        """Test compiled expression with single child returns child directly."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_compiled_node
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledExpression,
            CompiledValue,
            ConditionOperator,
            LogicalOperator,
            ValueType,
        )

        # Create expression with single child
        child = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="active", field_path=()),
            original="document.status == 'active'"
        )
        expr = CompiledExpression(
            operator=LogicalOperator.OR,
            children=[child],
            original="document.status == 'active'"
        )

        result, params = _build_pgvector_from_compiled_node(expr, {"id": "alice"})
        assert "status" in result
        assert "active" in params

    def test_compiled_expression_and_operator(self):
        """Test compiled expression with AND operator."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_compiled_node
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledExpression,
            CompiledValue,
            ConditionOperator,
            LogicalOperator,
            ValueType,
        )

        child1 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="active", field_path=()),
            original="document.status == 'active'"
        )
        child2 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("type",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="doc", field_path=()),
            original="document.type == 'doc'"
        )
        expr = CompiledExpression(
            operator=LogicalOperator.AND,
            children=[child1, child2],
            original="(document.status == 'active') AND (document.type == 'doc')"
        )

        result, params = _build_pgvector_from_compiled_node(expr, {"id": "alice"})
        assert " AND " in result

    def test_compiled_expression_unknown_type(self):
        """Test compiled expression with unknown type returns empty."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_compiled_node

        # Pass a non-expression/condition object
        result, params = _build_pgvector_from_compiled_node("not an expression", {"id": "alice"})
        assert result == ""
        assert params == []


class TestPgvectorFromCondition:
    """Tests for _build_pgvector_from_condition with various operators."""

    def test_equals_user_field_to_doc_field_bool(self):
        """Test EQUALS with user field to document field where user value is boolean."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        condition = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.USER_FIELD, value=None, field_path=("is_admin",)),
            right=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("admin_only",)),
            original="user.is_admin == document.admin_only"
        )

        result, params = _build_pgvector_from_condition(condition, {"id": "alice", "is_admin": True})
        assert "::boolean" in result
        assert True in params

    def test_equals_user_field_to_doc_field_numeric(self):
        """Test EQUALS with user field to document field where user value is numeric."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        condition = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.USER_FIELD, value=None, field_path=("level",)),
            right=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("required_level",)),
            original="user.level == document.required_level"
        )

        result, params = _build_pgvector_from_condition(condition, {"id": "alice", "level": 5})
        assert "::numeric" in result
        assert 5 in params

    def test_equals_user_field_to_literal_match(self):
        """Test EQUALS with user field to literal that matches."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        condition = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.USER_FIELD, value=None, field_path=("role",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="admin", field_path=()),
            original="user.role == 'admin'"
        )

        result, params = _build_pgvector_from_condition(condition, {"id": "alice", "role": "admin"})
        assert result == "TRUE"
        assert params == []

    def test_equals_user_field_to_literal_no_match(self):
        """Test EQUALS with user field to literal that doesn't match."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        condition = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.USER_FIELD, value=None, field_path=("role",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="admin", field_path=()),
            original="user.role == 'admin'"
        )

        result, params = _build_pgvector_from_condition(condition, {"id": "alice", "role": "user"})
        assert result == "FALSE"
        assert params == []

    def test_equals_doc_field_to_bool_literal(self):
        """Test EQUALS with document field to boolean literal."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        condition = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("active",)),
            right=CompiledValue(value_type=ValueType.LITERAL_BOOL, value=True, field_path=()),
            original="document.active == true"
        )

        result, params = _build_pgvector_from_condition(condition, {"id": "alice"})
        assert "::boolean" in result
        assert True in params

    def test_equals_doc_field_to_numeric_literal(self):
        """Test EQUALS with document field to numeric literal."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        condition = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("count",)),
            right=CompiledValue(value_type=ValueType.LITERAL_NUMBER, value=10, field_path=()),
            original="document.count == 10"
        )

        result, params = _build_pgvector_from_condition(condition, {"id": "alice"})
        assert "::numeric" in result
        assert 10 in params

    def test_not_equals_doc_field_bool(self):
        """Test NOT_EQUALS with document field and boolean."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        condition = CompiledCondition(
            operator=ConditionOperator.NOT_EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("active",)),
            right=CompiledValue(value_type=ValueType.LITERAL_BOOL, value=False, field_path=()),
            original="document.active != false"
        )

        result, params = _build_pgvector_from_condition(condition, {"id": "alice"})
        assert "::boolean" in result
        assert "!=" in result

    def test_not_equals_doc_field_numeric(self):
        """Test NOT_EQUALS with document field and numeric."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        condition = CompiledCondition(
            operator=ConditionOperator.NOT_EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("count",)),
            right=CompiledValue(value_type=ValueType.LITERAL_NUMBER, value=0, field_path=()),
            original="document.count != 0"
        )

        result, params = _build_pgvector_from_condition(condition, {"id": "alice"})
        assert "::numeric" in result
        assert "!=" in result

    def test_in_doc_field_empty_list(self):
        """Test IN with document field and empty list returns FALSE."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        condition = CompiledCondition(
            operator=ConditionOperator.IN,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("category",)),
            right=CompiledValue(value_type=ValueType.LITERAL_LIST, value=[], field_path=()),
            original="document.category in []"
        )

        result, params = _build_pgvector_from_condition(condition, {"id": "alice"})
        assert result == "FALSE"

    def test_in_literal_in_doc_array(self):
        """Test IN with literal string in document array field."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        condition = CompiledCondition(
            operator=ConditionOperator.IN,
            left=CompiledValue(value_type=ValueType.LITERAL_STRING, value="public", field_path=()),
            right=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("tags",)),
            original="'public' in document.tags"
        )

        result, params = _build_pgvector_from_condition(condition, {"id": "alice"})
        assert "jsonb_array_elements_text" in result
        assert "public" in params

    def test_not_in_user_field_none_value(self):
        """Test NOT_IN with user field that is None returns TRUE."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        condition = CompiledCondition(
            operator=ConditionOperator.NOT_IN,
            left=CompiledValue(value_type=ValueType.USER_FIELD, value=None, field_path=("tag",)),
            right=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("allowed_tags",)),
            original="user.tag not in document.allowed_tags"
        )

        result, params = _build_pgvector_from_condition(condition, {"id": "alice"})  # No tag field
        assert result == "TRUE"

    def test_not_in_literal_not_in_doc_array(self):
        """Test NOT_IN with literal not in document array field."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        condition = CompiledCondition(
            operator=ConditionOperator.NOT_IN,
            left=CompiledValue(value_type=ValueType.LITERAL_STRING, value="archived", field_path=()),
            right=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("tags",)),
            original="'archived' not in document.tags"
        )

        result, params = _build_pgvector_from_condition(condition, {"id": "alice"})
        assert "NOT" in result
        assert "jsonb_array_elements_text" in result
        assert "archived" in params

    def test_greater_than_user_vs_literal_true(self):
        """Test GREATER_THAN with user field vs literal that is true."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        condition = CompiledCondition(
            operator=ConditionOperator.GREATER_THAN,
            left=CompiledValue(value_type=ValueType.USER_FIELD, value=None, field_path=("level",)),
            right=CompiledValue(value_type=ValueType.LITERAL_NUMBER, value=3, field_path=()),
            original="user.level > 3"
        )

        result, params = _build_pgvector_from_condition(condition, {"id": "alice", "level": 5})
        assert result == "TRUE"

    def test_greater_than_user_vs_literal_false(self):
        """Test GREATER_THAN with user field vs literal that is false."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        condition = CompiledCondition(
            operator=ConditionOperator.GREATER_THAN,
            left=CompiledValue(value_type=ValueType.USER_FIELD, value=None, field_path=("level",)),
            right=CompiledValue(value_type=ValueType.LITERAL_NUMBER, value=10, field_path=()),
            original="user.level > 10"
        )

        result, params = _build_pgvector_from_condition(condition, {"id": "alice", "level": 5})
        assert result == "FALSE"

    def test_greater_than_or_equal_user_vs_literal(self):
        """Test GREATER_THAN_OR_EQUAL with user field vs literal."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        condition = CompiledCondition(
            operator=ConditionOperator.GREATER_THAN_OR_EQUAL,
            left=CompiledValue(value_type=ValueType.USER_FIELD, value=None, field_path=("level",)),
            right=CompiledValue(value_type=ValueType.LITERAL_NUMBER, value=5, field_path=()),
            original="user.level >= 5"
        )

        result, params = _build_pgvector_from_condition(condition, {"id": "alice", "level": 5})
        assert result == "TRUE"

    def test_less_than_user_vs_literal(self):
        """Test LESS_THAN with user field vs literal."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        condition = CompiledCondition(
            operator=ConditionOperator.LESS_THAN,
            left=CompiledValue(value_type=ValueType.USER_FIELD, value=None, field_path=("level",)),
            right=CompiledValue(value_type=ValueType.LITERAL_NUMBER, value=10, field_path=()),
            original="user.level < 10"
        )

        result, params = _build_pgvector_from_condition(condition, {"id": "alice", "level": 5})
        assert result == "TRUE"

    def test_less_than_or_equal_user_vs_literal(self):
        """Test LESS_THAN_OR_EQUAL with user field vs literal."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        condition = CompiledCondition(
            operator=ConditionOperator.LESS_THAN_OR_EQUAL,
            left=CompiledValue(value_type=ValueType.USER_FIELD, value=None, field_path=("level",)),
            right=CompiledValue(value_type=ValueType.LITERAL_NUMBER, value=5, field_path=()),
            original="user.level <= 5"
        )

        result, params = _build_pgvector_from_condition(condition, {"id": "alice", "level": 5})
        assert result == "TRUE"


class TestPgvectorLegacyConditionParsing:
    """Tests for legacy string-based condition parsing."""

    def test_legacy_not_exists_condition(self):
        """Test legacy 'not exists' condition parsing."""
        from ragguard.filters.backends.pgvector import _parse_pgvector_legacy_condition

        result, params = _parse_pgvector_legacy_condition(
            "document.deleted not exists",
            {"id": "alice"}
        )
        assert "IS NULL" in result
        assert "deleted" in result

    def test_legacy_exists_condition(self):
        """Test legacy 'exists' condition parsing."""
        from ragguard.filters.backends.pgvector import _parse_pgvector_legacy_condition

        result, params = _parse_pgvector_legacy_condition(
            "document.title exists",
            {"id": "alice"}
        )
        assert "IS NOT NULL" in result
        assert "title" in result

    def test_legacy_not_exists_user_field(self):
        """Test legacy 'not exists' with user field returns empty."""
        from ragguard.filters.backends.pgvector import _parse_pgvector_legacy_condition

        result, params = _parse_pgvector_legacy_condition(
            "user.role not exists",
            {"id": "alice"}
        )
        assert result == ""

    def test_legacy_exists_user_field(self):
        """Test legacy 'exists' with user field returns empty."""
        from ragguard.filters.backends.pgvector import _parse_pgvector_legacy_condition

        result, params = _parse_pgvector_legacy_condition(
            "user.role exists",
            {"id": "alice"}
        )
        assert result == ""

    def test_legacy_not_equals_document_field(self):
        """Test legacy != with document field."""
        from ragguard.filters.backends.pgvector import _parse_pgvector_legacy_condition

        result, params = _parse_pgvector_legacy_condition(
            "document.status != 'archived'",
            {"id": "alice"}
        )
        assert "!=" in result
        assert "archived" in params

    def test_legacy_not_in_user_field_none(self):
        """Test legacy not in with user field that is None."""
        from ragguard.filters.backends.pgvector import _parse_pgvector_legacy_condition

        result, params = _parse_pgvector_legacy_condition(
            "user.missing not in document.tags",
            {"id": "alice"}  # No 'missing' field
        )
        assert "1 = 1" in result  # None value means always true for not in

    def test_legacy_not_in_literal_in_doc(self):
        """Test legacy not in with literal value."""
        from ragguard.filters.backends.pgvector import _parse_pgvector_legacy_condition

        result, params = _parse_pgvector_legacy_condition(
            "'archived' not in document.tags",
            {"id": "alice"}
        )
        assert "NOT" in result
        assert "archived" in params

    def test_legacy_not_in_doc_in_list(self):
        """Test legacy not in with document field in list."""
        from ragguard.filters.backends.pgvector import _parse_pgvector_legacy_condition

        result, params = _parse_pgvector_legacy_condition(
            "document.status not in ['deleted', 'archived']",
            {"id": "alice"}
        )
        assert "NOT IN" in result
        assert "deleted" in params
        assert "archived" in params

    def test_legacy_not_in_doc_in_empty_list(self):
        """Test legacy not in with empty list returns always true."""
        from ragguard.filters.backends.pgvector import _parse_pgvector_legacy_condition

        result, params = _parse_pgvector_legacy_condition(
            "document.status not in []",
            {"id": "alice"}
        )
        assert "1 = 1" in result

    def test_legacy_in_user_field_in_doc(self):
        """Test legacy in with user field in document array."""
        from ragguard.filters.backends.pgvector import _parse_pgvector_legacy_condition

        result, params = _parse_pgvector_legacy_condition(
            "user.role in document.allowed_roles",
            {"id": "alice", "role": "admin"}
        )
        assert "ANY" in result
        assert "admin" in params

    def test_legacy_in_literal_in_doc(self):
        """Test legacy in with literal in document array."""
        from ragguard.filters.backends.pgvector import _parse_pgvector_legacy_condition

        result, params = _parse_pgvector_legacy_condition(
            "'public' in document.tags",
            {"id": "alice"}
        )
        assert "ANY" in result
        assert "public" in params

    def test_legacy_in_doc_in_list(self):
        """Test legacy in with document field in list."""
        from ragguard.filters.backends.pgvector import _parse_pgvector_legacy_condition

        result, params = _parse_pgvector_legacy_condition(
            "document.status in ['active', 'pending']",
            {"id": "alice"}
        )
        assert "IN" in result
        assert "active" in params
        assert "pending" in params

    def test_legacy_in_doc_in_empty_list(self):
        """Test legacy in with document field in empty list returns false."""
        from ragguard.filters.backends.pgvector import _parse_pgvector_legacy_condition

        result, params = _parse_pgvector_legacy_condition(
            "document.status in []",
            {"id": "alice"}
        )
        assert "1 = 0" in result

    def test_legacy_unrecognized_condition(self):
        """Test legacy condition that doesn't match any pattern."""
        from ragguard.filters.backends.pgvector import _parse_pgvector_legacy_condition

        result, params = _parse_pgvector_legacy_condition(
            "some.unknown.condition",
            {"id": "alice"}
        )
        assert result == ""


class TestPgvectorConditionFilterErrors:
    """Test error handling in _build_pgvector_condition_filter."""

    def test_invalid_condition_returns_empty(self):
        """Test that invalid conditions are skipped gracefully."""
        from ragguard.filters.backends.pgvector import _build_pgvector_condition_filter

        # Should not raise, but return empty for unparseable condition
        result, params = _build_pgvector_condition_filter(
            "this is not a valid condition format",
            {"id": "alice"}
        )
        assert result == ""


class TestPgvectorUnsupportedCondition:
    """Test unsupported conditions raise appropriate errors."""

    def test_unsupported_operator_combination_raises(self):
        """Test that unsupported operator/type combinations raise error."""
        from ragguard.exceptions import UnsupportedConditionError
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        # Create a condition with an unusual combination - EXISTS with a right side
        condition = CompiledCondition(
            operator=ConditionOperator.EXISTS,
            left=CompiledValue(value_type=ValueType.USER_FIELD, value=None, field_path=("role",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="admin", field_path=()),
            original="user.role exists 'admin'"
        )

        with pytest.raises(UnsupportedConditionError, match="pgvector"):
            _build_pgvector_from_condition(condition, {"id": "alice", "role": "admin"})


class TestPgvectorLiteralNone:
    """Test handling of None literal values."""

    def test_literal_none_in_condition(self):
        """Test condition with LITERAL_NONE value type."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        condition = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("deleted",)),
            right=CompiledValue(value_type=ValueType.LITERAL_NONE, value=None, field_path=()),
            original="document.deleted == null"
        )

        result, params = _build_pgvector_from_condition(condition, {"id": "alice"})
        # Should handle null comparison - returns equality to NULL
        assert "deleted" in result


class TestPgvectorFieldAccessor:
    """Test JSONB field accessor generation."""

    def test_field_accessor_without_as_text(self):
        """Test JSONB accessor without as_text conversion."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        # IN operator accesses array field without as_text
        condition = CompiledCondition(
            operator=ConditionOperator.IN,
            left=CompiledValue(value_type=ValueType.USER_FIELD, value=None, field_path=("id",)),
            right=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("user_ids",)),
            original="user.id in document.user_ids"
        )

        result, params = _build_pgvector_from_condition(condition, {"id": "alice"})
        assert "jsonb_array_elements_text" in result


class TestPgvectorExistsOperators:
    """Tests for EXISTS and NOT_EXISTS operators."""

    def test_exists_operator_document_field(self):
        """Test EXISTS operator with document field."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        condition = CompiledCondition(
            operator=ConditionOperator.EXISTS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("title",)),
            right=None,
            original="document.title exists"
        )

        result, params = _build_pgvector_from_condition(condition, {"id": "alice"})
        assert "IS NOT NULL" in result
        assert "title" in result

    def test_not_exists_operator_document_field(self):
        """Test NOT_EXISTS operator with document field."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        condition = CompiledCondition(
            operator=ConditionOperator.NOT_EXISTS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("deleted",)),
            right=None,
            original="document.deleted not exists"
        )

        result, params = _build_pgvector_from_condition(condition, {"id": "alice"})
        assert "IS NULL" in result
        assert "deleted" in result


class TestPgvectorToPgvectorFilter:
    """Tests for the main to_pgvector_filter function."""

    def test_filter_with_match_conditions(self):
        """Test building filter with match conditions."""
        from ragguard import Policy
        from ragguard.filters.backends.pgvector import to_pgvector_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "dept_rule",
                "match": {"department": "engineering"},
                "allow": {"everyone": True}
            }],
            "default": "deny"
        })

        where_clause, params = to_pgvector_filter(policy, {"id": "alice"})
        assert "department" in where_clause
        assert "engineering" in params

    def test_filter_with_match_list(self):
        """Test building filter with match condition containing list."""
        from ragguard import Policy
        from ragguard.filters.backends.pgvector import to_pgvector_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "dept_rule",
                "match": {"department": ["engineering", "product"]},
                "allow": {"everyone": True}
            }],
            "default": "deny"
        })

        where_clause, params = to_pgvector_filter(policy, {"id": "alice"})
        assert "department" in where_clause
        assert "IN" in where_clause
        assert "engineering" in params
        assert "product" in params

    def test_filter_default_allow_no_rules(self):
        """Test filter when no rules apply and default is allow."""
        from ragguard import Policy
        from ragguard.filters.backends.pgvector import to_pgvector_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "admin_rule",
                "allow": {"conditions": ["user.role == 'admin'"]}
            }],
            "default": "allow"
        })

        # Non-admin user, rule doesn't apply
        where_clause, params = to_pgvector_filter(policy, {"id": "alice", "role": "user"})
        # Default is allow, so no filter needed (empty clause or TRUE)

    def test_filter_default_deny_no_rules(self):
        """Test filter when no rules apply and default is deny."""
        from ragguard import Policy
        from ragguard.filters.backends.pgvector import to_pgvector_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "admin_rule",
                "allow": {"conditions": ["user.role == 'admin'"]}
            }],
            "default": "deny"
        })

        # Non-admin user, rule doesn't apply
        where_clause, params = to_pgvector_filter(policy, {"id": "alice", "role": "user"})
        assert "FALSE" in where_clause
