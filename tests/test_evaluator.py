"""
Comprehensive tests for condition evaluator to maximize coverage.
"""

import pytest

from ragguard.policy.compiler.evaluator import CompiledConditionEvaluator
from ragguard.policy.compiler.models import (
    CompiledCondition,
    CompiledExpression,
    CompiledValue,
    ConditionOperator,
    ConditionType,
    LogicalOperator,
    ValueType,
)


def make_user_field(path: tuple) -> CompiledValue:
    """Helper to create user field CompiledValue."""
    return CompiledValue(value_type=ValueType.USER_FIELD, value=None, field_path=path)


def make_doc_field(path: tuple) -> CompiledValue:
    """Helper to create document field CompiledValue."""
    return CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=path)


def make_literal_str(val: str) -> CompiledValue:
    """Helper to create literal string CompiledValue."""
    return CompiledValue(value_type=ValueType.LITERAL_STRING, value=val, field_path=())


def make_literal_num(val) -> CompiledValue:
    """Helper to create literal number CompiledValue."""
    return CompiledValue(value_type=ValueType.LITERAL_NUMBER, value=val, field_path=())


def make_literal_bool(val: bool) -> CompiledValue:
    """Helper to create literal bool CompiledValue."""
    return CompiledValue(value_type=ValueType.LITERAL_BOOL, value=val, field_path=())


def make_literal_list(val: list) -> CompiledValue:
    """Helper to create literal list CompiledValue."""
    return CompiledValue(value_type=ValueType.LITERAL_LIST, value=val, field_path=())


def make_literal_none() -> CompiledValue:
    """Helper to create literal None CompiledValue."""
    return CompiledValue(value_type=ValueType.LITERAL_NONE, value=None, field_path=())


class TestEvaluateNode:
    """Tests for evaluate_node method."""

    def test_evaluate_simple_condition(self):
        """Test evaluating a simple condition."""
        condition = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=make_user_field(("department",)),
            right=make_doc_field(("department",)),
            original="user.department == document.department"
        )

        user = {"department": "engineering"}
        document = {"department": "engineering"}

        result = CompiledConditionEvaluator.evaluate_node(condition, user, document)
        assert result is True

    def test_evaluate_expression(self):
        """Test evaluating an expression."""
        condition1 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=make_doc_field(("visibility",)),
            right=make_literal_str("public"),
            original="document.visibility == 'public'"
        )
        condition2 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=make_user_field(("role",)),
            right=make_literal_str("admin"),
            original="user.role == 'admin'"
        )

        expr = CompiledExpression(
            operator=LogicalOperator.OR,
            children=[condition1, condition2],
            original="cond1 OR cond2"
        )

        user = {"role": "user"}
        document = {"visibility": "public"}

        result = CompiledConditionEvaluator.evaluate_node(expr, user, document)
        assert result is True

    def test_evaluate_unknown_node_type(self):
        """Test evaluating unknown node type raises error."""
        with pytest.raises(ValueError, match="Unknown node type"):
            CompiledConditionEvaluator.evaluate_node("invalid", {}, {})


class TestEvaluateExpression:
    """Tests for evaluate_expression method."""

    def test_and_expression_all_true(self):
        """Test AND expression with all conditions true."""
        cond1 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=make_user_field(("dept",)),
            right=make_literal_str("eng"),
            original="user.dept == 'eng'"
        )
        cond2 = CompiledCondition(
            operator=ConditionOperator.GREATER_THAN,
            left=make_user_field(("level",)),
            right=make_literal_num(3),
            original="user.level > 3"
        )

        expr = CompiledExpression(
            operator=LogicalOperator.AND,
            children=[cond1, cond2],
            original="cond1 AND cond2"
        )

        result = CompiledConditionEvaluator.evaluate_expression(
            expr, {"dept": "eng", "level": 5}, {}
        )
        assert result is True

    def test_and_expression_one_false(self):
        """Test AND expression with one condition false."""
        cond1 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=make_user_field(("dept",)),
            right=make_literal_str("eng"),
            original="user.dept == 'eng'"
        )
        cond2 = CompiledCondition(
            operator=ConditionOperator.GREATER_THAN,
            left=make_user_field(("level",)),
            right=make_literal_num(10),
            original="user.level > 10"
        )

        expr = CompiledExpression(
            operator=LogicalOperator.AND,
            children=[cond1, cond2],
            original="cond1 AND cond2"
        )

        result = CompiledConditionEvaluator.evaluate_expression(
            expr, {"dept": "eng", "level": 5}, {}
        )
        assert result is False

    def test_or_expression_one_true(self):
        """Test OR expression with one condition true."""
        cond1 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=make_user_field(("role",)),
            right=make_literal_str("admin"),
            original="user.role == 'admin'"
        )
        cond2 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=make_doc_field(("public",)),
            right=make_literal_bool(True),
            original="document.public == true"
        )

        expr = CompiledExpression(
            operator=LogicalOperator.OR,
            children=[cond1, cond2],
            original="cond1 OR cond2"
        )

        result = CompiledConditionEvaluator.evaluate_expression(
            expr, {"role": "user"}, {"public": True}
        )
        assert result is True

    def test_or_expression_all_false(self):
        """Test OR expression with all conditions false."""
        cond1 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=make_user_field(("role",)),
            right=make_literal_str("admin"),
            original="user.role == 'admin'"
        )
        cond2 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=make_doc_field(("public",)),
            right=make_literal_bool(True),
            original="document.public == true"
        )

        expr = CompiledExpression(
            operator=LogicalOperator.OR,
            children=[cond1, cond2],
            original="cond1 OR cond2"
        )

        result = CompiledConditionEvaluator.evaluate_expression(
            expr, {"role": "user"}, {"public": False}
        )
        assert result is False


class TestEvaluateConditionOperators:
    """Tests for various condition operators."""

    def test_exists_operator_true(self):
        """Test EXISTS operator when field exists."""
        condition = CompiledCondition(
            operator=ConditionOperator.EXISTS,
            left=make_user_field(("department",)),
            right=make_literal_none(),
            original="EXISTS(user.department)"
        )

        result = CompiledConditionEvaluator.evaluate(
            condition, {"department": "eng"}, {}
        )
        assert result is True

    def test_exists_operator_false(self):
        """Test EXISTS operator when field is missing."""
        condition = CompiledCondition(
            operator=ConditionOperator.EXISTS,
            left=make_user_field(("department",)),
            right=make_literal_none(),
            original="EXISTS(user.department)"
        )

        result = CompiledConditionEvaluator.evaluate(condition, {}, {})
        assert result is False

    def test_not_exists_operator_true(self):
        """Test NOT_EXISTS operator when field is missing."""
        condition = CompiledCondition(
            operator=ConditionOperator.NOT_EXISTS,
            left=make_user_field(("temp",)),
            right=make_literal_none(),
            original="NOT EXISTS(user.temp)"
        )

        result = CompiledConditionEvaluator.evaluate(condition, {}, {})
        assert result is True

    def test_not_exists_operator_false(self):
        """Test NOT_EXISTS operator when field exists."""
        condition = CompiledCondition(
            operator=ConditionOperator.NOT_EXISTS,
            left=make_user_field(("temp",)),
            right=make_literal_none(),
            original="NOT EXISTS(user.temp)"
        )

        result = CompiledConditionEvaluator.evaluate(condition, {"temp": "value"}, {})
        assert result is False

    def test_greater_than_true(self):
        """Test GREATER_THAN operator when true."""
        condition = CompiledCondition(
            operator=ConditionOperator.GREATER_THAN,
            left=make_user_field(("level",)),
            right=make_literal_num(5),
            original="user.level > 5"
        )

        result = CompiledConditionEvaluator.evaluate(condition, {"level": 10}, {})
        assert result is True

    def test_greater_than_false(self):
        """Test GREATER_THAN operator when false."""
        condition = CompiledCondition(
            operator=ConditionOperator.GREATER_THAN,
            left=make_user_field(("level",)),
            right=make_literal_num(5),
            original="user.level > 5"
        )

        result = CompiledConditionEvaluator.evaluate(condition, {"level": 3}, {})
        assert result is False

    def test_greater_than_with_none(self):
        """Test GREATER_THAN with None value."""
        condition = CompiledCondition(
            operator=ConditionOperator.GREATER_THAN,
            left=make_user_field(("level",)),
            right=make_literal_num(5),
            original="user.level > 5"
        )

        result = CompiledConditionEvaluator.evaluate(condition, {}, {})
        assert result is False

    def test_greater_than_type_mismatch(self):
        """Test GREATER_THAN with type mismatch."""
        condition = CompiledCondition(
            operator=ConditionOperator.GREATER_THAN,
            left=make_user_field(("level",)),
            right=make_literal_str("five"),
            original="user.level > 'five'"
        )

        result = CompiledConditionEvaluator.evaluate(condition, {"level": 10}, {})
        assert result is False

    def test_less_than_true(self):
        """Test LESS_THAN operator when true."""
        condition = CompiledCondition(
            operator=ConditionOperator.LESS_THAN,
            left=make_user_field(("level",)),
            right=make_literal_num(10),
            original="user.level < 10"
        )

        result = CompiledConditionEvaluator.evaluate(condition, {"level": 5}, {})
        assert result is True

    def test_less_than_with_none(self):
        """Test LESS_THAN with None value."""
        condition = CompiledCondition(
            operator=ConditionOperator.LESS_THAN,
            left=make_user_field(("level",)),
            right=make_literal_num(5),
            original="user.level < 5"
        )

        result = CompiledConditionEvaluator.evaluate(condition, {}, {})
        assert result is False

    def test_greater_than_or_equal_true(self):
        """Test GREATER_THAN_OR_EQUAL operator."""
        condition = CompiledCondition(
            operator=ConditionOperator.GREATER_THAN_OR_EQUAL,
            left=make_user_field(("level",)),
            right=make_literal_num(5),
            original="user.level >= 5"
        )

        result = CompiledConditionEvaluator.evaluate(condition, {"level": 5}, {})
        assert result is True

    def test_greater_than_or_equal_with_none(self):
        """Test GREATER_THAN_OR_EQUAL with None."""
        condition = CompiledCondition(
            operator=ConditionOperator.GREATER_THAN_OR_EQUAL,
            left=make_user_field(("level",)),
            right=make_literal_num(5),
            original="user.level >= 5"
        )

        result = CompiledConditionEvaluator.evaluate(condition, {}, {})
        assert result is False

    def test_less_than_or_equal_true(self):
        """Test LESS_THAN_OR_EQUAL operator."""
        condition = CompiledCondition(
            operator=ConditionOperator.LESS_THAN_OR_EQUAL,
            left=make_user_field(("level",)),
            right=make_literal_num(5),
            original="user.level <= 5"
        )

        result = CompiledConditionEvaluator.evaluate(condition, {"level": 5}, {})
        assert result is True

    def test_less_than_or_equal_with_none(self):
        """Test LESS_THAN_OR_EQUAL with None."""
        condition = CompiledCondition(
            operator=ConditionOperator.LESS_THAN_OR_EQUAL,
            left=make_user_field(("level",)),
            right=make_literal_num(5),
            original="user.level <= 5"
        )

        result = CompiledConditionEvaluator.evaluate(condition, {}, {})
        assert result is False

    def test_equals_true(self):
        """Test EQUALS operator when true."""
        condition = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=make_user_field(("dept",)),
            right=make_doc_field(("dept",)),
            original="user.dept == document.dept"
        )

        result = CompiledConditionEvaluator.evaluate(
            condition, {"dept": "eng"}, {"dept": "eng"}
        )
        assert result is True

    def test_equals_with_none(self):
        """Test EQUALS with None value denies access."""
        condition = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=make_user_field(("dept",)),
            right=make_doc_field(("dept",)),
            original="user.dept == document.dept"
        )

        result = CompiledConditionEvaluator.evaluate(condition, {}, {"dept": "eng"})
        assert result is False

    def test_not_equals_true(self):
        """Test NOT_EQUALS operator when true."""
        condition = CompiledCondition(
            operator=ConditionOperator.NOT_EQUALS,
            left=make_user_field(("dept",)),
            right=make_literal_str("hr"),
            original="user.dept != 'hr'"
        )

        result = CompiledConditionEvaluator.evaluate(condition, {"dept": "eng"}, {})
        assert result is True

    def test_not_equals_with_none(self):
        """Test NOT_EQUALS with None value denies access."""
        condition = CompiledCondition(
            operator=ConditionOperator.NOT_EQUALS,
            left=make_user_field(("dept",)),
            right=make_literal_str("hr"),
            original="user.dept != 'hr'"
        )

        result = CompiledConditionEvaluator.evaluate(condition, {}, {})
        assert result is False

    def test_in_operator_value_in_list(self):
        """Test IN operator with value in list."""
        condition = CompiledCondition(
            operator=ConditionOperator.IN,
            left=make_user_field(("group",)),
            right=make_doc_field(("allowed_groups",)),
            original="user.group in document.allowed_groups",
            condition_type=ConditionType.VALUE_IN_ARRAY_FIELD
        )

        result = CompiledConditionEvaluator.evaluate(
            condition,
            {"group": "eng"},
            {"allowed_groups": ["eng", "sales", "hr"]}
        )
        assert result is True

    def test_in_operator_value_not_in_list(self):
        """Test IN operator with value not in list."""
        condition = CompiledCondition(
            operator=ConditionOperator.IN,
            left=make_user_field(("group",)),
            right=make_doc_field(("allowed_groups",)),
            original="user.group in document.allowed_groups",
            condition_type=ConditionType.VALUE_IN_ARRAY_FIELD
        )

        result = CompiledConditionEvaluator.evaluate(
            condition,
            {"group": "finance"},
            {"allowed_groups": ["eng", "sales", "hr"]}
        )
        assert result is False

    def test_in_operator_with_non_list(self):
        """Test IN operator when right is not a list."""
        condition = CompiledCondition(
            operator=ConditionOperator.IN,
            left=make_user_field(("group",)),
            right=make_doc_field(("group",)),
            original="user.group in document.group",
            condition_type=ConditionType.VALUE_IN_ARRAY_FIELD
        )

        result = CompiledConditionEvaluator.evaluate(
            condition,
            {"group": "eng"},
            {"group": "eng"}  # Not a list
        )
        assert result is False

    def test_not_in_operator_true(self):
        """Test NOT_IN operator when true."""
        condition = CompiledCondition(
            operator=ConditionOperator.NOT_IN,
            left=make_user_field(("dept",)),
            right=make_literal_list(["hr", "finance"]),
            original="user.dept not in ['hr', 'finance']"
        )

        result = CompiledConditionEvaluator.evaluate(
            condition, {"dept": "eng"}, {}
        )
        assert result is True

    def test_not_in_operator_false(self):
        """Test NOT_IN operator when false."""
        condition = CompiledCondition(
            operator=ConditionOperator.NOT_IN,
            left=make_user_field(("dept",)),
            right=make_literal_list(["eng", "sales"]),
            original="user.dept not in ['eng', 'sales']"
        )

        result = CompiledConditionEvaluator.evaluate(
            condition, {"dept": "eng"}, {}
        )
        assert result is False


class TestResolveValue:
    """Tests for _resolve_value method."""

    def test_resolve_literal_string(self):
        """Test resolving literal string."""
        value = make_literal_str("test")
        result = CompiledConditionEvaluator._resolve_value(value, {}, {})
        assert result == "test"

    def test_resolve_literal_number(self):
        """Test resolving literal number."""
        value = make_literal_num(42)
        result = CompiledConditionEvaluator._resolve_value(value, {}, {})
        assert result == 42

    def test_resolve_literal_bool(self):
        """Test resolving literal bool."""
        value = make_literal_bool(True)
        result = CompiledConditionEvaluator._resolve_value(value, {}, {})
        assert result is True

    def test_resolve_literal_none(self):
        """Test resolving literal None."""
        value = make_literal_none()
        result = CompiledConditionEvaluator._resolve_value(value, {}, {})
        assert result is None

    def test_resolve_literal_list(self):
        """Test resolving literal list."""
        value = make_literal_list(["a", "b", "c"])
        result = CompiledConditionEvaluator._resolve_value(value, {}, {})
        assert result == ["a", "b", "c"]

    def test_resolve_user_field(self):
        """Test resolving user field."""
        value = make_user_field(("department",))
        result = CompiledConditionEvaluator._resolve_value(
            value, {"department": "engineering"}, {}
        )
        assert result == "engineering"

    def test_resolve_document_field(self):
        """Test resolving document field."""
        value = make_doc_field(("type",))
        result = CompiledConditionEvaluator._resolve_value(
            value, {}, {"type": "public"}
        )
        assert result == "public"

    def test_resolve_nested_user_field(self):
        """Test resolving nested user field."""
        value = make_user_field(("org", "department"))
        result = CompiledConditionEvaluator._resolve_value(
            value, {"org": {"department": "eng"}}, {}
        )
        assert result == "eng"

    def test_resolve_missing_nested_field(self):
        """Test resolving missing nested field returns None."""
        value = make_user_field(("org", "department"))
        result = CompiledConditionEvaluator._resolve_value(value, {"org": {}}, {})
        assert result is None


class TestGetNestedValue:
    """Tests for _get_nested_value method."""

    def test_get_simple_value(self):
        """Test getting simple value."""
        result = CompiledConditionEvaluator._get_nested_value(
            {"name": "test"}, ("name",)
        )
        assert result == "test"

    def test_get_nested_value(self):
        """Test getting nested value."""
        result = CompiledConditionEvaluator._get_nested_value(
            {"org": {"team": {"name": "ML"}}},
            ("org", "team", "name")
        )
        assert result == "ML"

    def test_get_missing_key(self):
        """Test getting missing key returns None."""
        result = CompiledConditionEvaluator._get_nested_value(
            {"name": "test"}, ("missing",)
        )
        assert result is None

    def test_get_from_non_dict(self):
        """Test getting from non-dict returns None."""
        result = CompiledConditionEvaluator._get_nested_value(
            {"name": "test"}, ("name", "sub")
        )
        assert result is None
