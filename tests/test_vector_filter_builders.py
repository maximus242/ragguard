"""
Extended tests for vector database filter builders.

Tests string parsing fallback paths and compiled expression handling
for Milvus, pgvector, Pinecone, and Weaviate filter backends.
"""

import pytest

from ragguard.policy import Policy


def make_policy(rules):
    """Helper to create a policy from rules."""
    return Policy.from_dict({
        "version": "1",
        "rules": [{"name": f"rule_{i}", "allow": r} for i, r in enumerate(rules)],
        "default": "deny"
    })


# ============================================================
# Milvus String Parsing Tests
# ============================================================

class TestMilvusStringParsing:
    """Tests for Milvus string parsing fallback paths."""

    def test_parse_user_equals_document(self):
        """Test user.field == document.field parsing."""
        from ragguard.filters.backends.milvus import _build_milvus_condition_filter

        expr = _build_milvus_condition_filter(
            "user.department == document.department",
            {"id": "alice", "department": "engineering"}
        )
        assert expr is not None
        assert "department == 'engineering'" in expr

    def test_parse_user_equals_document_null(self):
        """Test user.field == document.field with null user value."""
        from ragguard.filters.backends.milvus import _build_milvus_condition_filter

        expr = _build_milvus_condition_filter(
            "user.department == document.department",
            {"id": "alice"}  # No department
        )
        assert expr == "id < 0"  # Deny all

    def test_parse_user_equals_document_bool(self):
        """Test user.field == document.field with boolean value."""
        from ragguard.filters.backends.milvus import _build_milvus_condition_filter

        expr = _build_milvus_condition_filter(
            "user.is_admin == document.requires_admin",
            {"id": "alice", "is_admin": True}
        )
        assert "requires_admin == true" in expr

    def test_parse_user_equals_document_number(self):
        """Test user.field == document.field with numeric value."""
        from ragguard.filters.backends.milvus import _build_milvus_condition_filter

        expr = _build_milvus_condition_filter(
            "user.level == document.required_level",
            {"id": "alice", "level": 5}
        )
        assert "required_level == 5" in expr

    def test_parse_document_equals_literal_string(self):
        """Test document.field == 'literal' parsing."""
        from ragguard.filters.backends.milvus import _build_milvus_condition_filter

        expr = _build_milvus_condition_filter(
            "document.status == 'active'",
            {"id": "alice"}
        )
        assert "status == 'active'" in expr

    def test_parse_document_equals_literal_bool(self):
        """Test document.field == true parsing."""
        from ragguard.filters.backends.milvus import _build_milvus_condition_filter

        expr = _build_milvus_condition_filter(
            "document.public == true",
            {"id": "alice"}
        )
        assert "public == true" in expr

    def test_parse_document_equals_literal_number(self):
        """Test document.field == 123 parsing."""
        from ragguard.filters.backends.milvus import _build_milvus_condition_filter

        expr = _build_milvus_condition_filter(
            "document.priority == 5",
            {"id": "alice"}
        )
        assert "priority == 5" in expr

    def test_parse_user_in_document(self):
        """Test user.field in document.array_field parsing."""
        from ragguard.filters.backends.milvus import _build_milvus_condition_filter

        expr = _build_milvus_condition_filter(
            "user.role in document.allowed_roles",
            {"id": "alice", "role": "admin"}
        )
        assert "array_contains" in expr
        assert "allowed_roles" in expr

    def test_parse_user_in_document_null(self):
        """Test user.field in document.array_field with null user value."""
        from ragguard.filters.backends.milvus import _build_milvus_condition_filter

        expr = _build_milvus_condition_filter(
            "user.role in document.allowed_roles",
            {"id": "alice"}  # No role
        )
        assert expr == "id < 0"  # Deny all

    def test_parse_invalid_condition(self):
        """Test invalid condition returns None."""
        from ragguard.filters.backends.milvus import _build_milvus_condition_filter

        expr = _build_milvus_condition_filter(
            "some random text",
            {"id": "alice"}
        )
        assert expr is None


class TestMilvusCompiledExpressions:
    """Tests for Milvus compiled expression handling."""

    def test_compiled_expression_or(self):
        """Test compiled OR expression."""
        from ragguard.filters.backends.milvus import _build_milvus_from_compiled_node
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledExpression,
            CompiledValue,
            ConditionOperator,
            LogicalOperator,
            ValueType,
        )

        cond1 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="active", field_path=()),
            original="test"
        )
        cond2 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="pending", field_path=()),
            original="test"
        )
        expr = CompiledExpression(operator=LogicalOperator.OR, children=[cond1, cond2], original="test")

        result = _build_milvus_from_compiled_node(expr, {"id": "alice"})

        assert "or" in result
        assert "status" in result

    def test_compiled_expression_and(self):
        """Test compiled AND expression."""
        from ragguard.filters.backends.milvus import _build_milvus_from_compiled_node
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledExpression,
            CompiledValue,
            ConditionOperator,
            LogicalOperator,
            ValueType,
        )

        cond1 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("public",)),
            right=CompiledValue(value_type=ValueType.LITERAL_BOOL, value=True, field_path=()),
            original="test"
        )
        cond2 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="active", field_path=()),
            original="test"
        )
        expr = CompiledExpression(operator=LogicalOperator.AND, children=[cond1, cond2], original="test")

        result = _build_milvus_from_compiled_node(expr, {"id": "alice"})

        assert "and" in result
        assert "public" in result
        assert "status" in result

    def test_compiled_expression_single_child(self):
        """Test compiled expression with single child."""
        from ragguard.filters.backends.milvus import _build_milvus_from_compiled_node
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledExpression,
            CompiledValue,
            ConditionOperator,
            LogicalOperator,
            ValueType,
        )

        cond1 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="active", field_path=()),
            original="test"
        )
        expr = CompiledExpression(operator=LogicalOperator.OR, children=[cond1], original="test")

        result = _build_milvus_from_compiled_node(expr, {"id": "alice"})

        # Single child, no OR/AND
        assert "or" not in result
        assert "status" in result

    def test_compiled_expression_empty(self):
        """Test compiled expression with no children."""
        from ragguard.filters.backends.milvus import _build_milvus_from_compiled_node
        from ragguard.policy.compiler import CompiledExpression, LogicalOperator

        expr = CompiledExpression(operator=LogicalOperator.OR, children=[], original="test")

        result = _build_milvus_from_compiled_node(expr, {"id": "alice"})

        assert result is None


# ============================================================
# pgvector String Parsing Tests
# ============================================================

class TestPgvectorStringParsing:
    """Tests for pgvector string parsing fallback paths."""

    def test_parse_user_equals_document(self):
        """Test user.field == document.field parsing."""
        from ragguard.filters.backends.pgvector import _build_pgvector_condition_filter

        clause, params = _build_pgvector_condition_filter(
            "user.department == document.department",
            {"id": "alice", "department": "engineering"}
        )
        assert "department" in clause
        assert "engineering" in params

    def test_parse_user_equals_document_null(self):
        """Test user.field == document.field with null user value."""
        from ragguard.filters.backends.pgvector import _build_pgvector_condition_filter

        clause, params = _build_pgvector_condition_filter(
            "user.department == document.department",
            {"id": "alice"}  # No department
        )
        assert clause == "FALSE"

    def test_parse_document_equals_literal(self):
        """Test document.field == 'literal' parsing."""
        from ragguard.filters.backends.pgvector import _build_pgvector_condition_filter

        clause, params = _build_pgvector_condition_filter(
            "document.status == 'active'",
            {"id": "alice"}
        )
        assert "status" in clause
        assert "active" in params

    def test_parse_user_in_document(self):
        """Test user.field in document.array_field parsing."""
        from ragguard.filters.backends.pgvector import _build_pgvector_condition_filter

        clause, params = _build_pgvector_condition_filter(
            "user.role in document.allowed_roles",
            {"id": "alice", "role": "admin"}
        )
        assert "allowed_roles" in clause
        assert "admin" in str(params)

    def test_parse_user_in_document_null(self):
        """Test user.field in document.array_field with null user value."""
        from ragguard.filters.backends.pgvector import _build_pgvector_condition_filter

        clause, params = _build_pgvector_condition_filter(
            "user.role in document.allowed_roles",
            {"id": "alice"}  # No role
        )
        assert clause == "FALSE"

    def test_parse_invalid_condition(self):
        """Test invalid condition returns empty clause."""
        from ragguard.filters.backends.pgvector import _build_pgvector_condition_filter

        result = _build_pgvector_condition_filter(
            "some random text",
            {"id": "alice"}
        )
        # pgvector returns empty clause for invalid conditions (skips them)
        assert result == ("", [])


class TestPgvectorCompiledExpressions:
    """Tests for pgvector compiled expression handling."""

    def test_compiled_expression_or(self):
        """Test compiled OR expression."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_compiled_node
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledExpression,
            CompiledValue,
            ConditionOperator,
            LogicalOperator,
            ValueType,
        )

        cond1 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="active", field_path=()),
            original="test"
        )
        cond2 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="pending", field_path=()),
            original="test"
        )
        expr = CompiledExpression(operator=LogicalOperator.OR, children=[cond1, cond2], original="test")

        clause, params = _build_pgvector_from_compiled_node(expr, {"id": "alice"})

        assert "OR" in clause
        assert "status" in clause

    def test_compiled_expression_and(self):
        """Test compiled AND expression."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_compiled_node
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledExpression,
            CompiledValue,
            ConditionOperator,
            LogicalOperator,
            ValueType,
        )

        cond1 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("public",)),
            right=CompiledValue(value_type=ValueType.LITERAL_BOOL, value=True, field_path=()),
            original="test"
        )
        cond2 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="active", field_path=()),
            original="test"
        )
        expr = CompiledExpression(operator=LogicalOperator.AND, children=[cond1, cond2], original="test")

        clause, params = _build_pgvector_from_compiled_node(expr, {"id": "alice"})

        assert "AND" in clause
        assert "public" in clause
        assert "status" in clause


# ============================================================
# Pinecone String Parsing Tests
# ============================================================

class TestPineconeStringParsing:
    """Tests for Pinecone string parsing fallback paths."""

    def test_parse_user_equals_document(self):
        """Test user.field == document.field parsing."""
        from ragguard.filters.backends.pinecone import _build_pinecone_condition_filter

        filter_dict = _build_pinecone_condition_filter(
            "user.department == document.department",
            {"id": "alice", "department": "engineering"}
        )
        assert filter_dict is not None
        assert "department" in filter_dict

    def test_parse_user_equals_document_null(self):
        """Test user.field == document.field with null user value returns deny-all sentinel."""
        from ragguard.filters.backends.pinecone import _build_pinecone_condition_filter

        filter_dict = _build_pinecone_condition_filter(
            "user.department == document.department",
            {"id": "alice"}  # No department
        )
        # Should return deny-all sentinel
        assert filter_dict is not None
        assert "__ragguard_deny_all__" in str(filter_dict)

    def test_parse_document_equals_literal(self):
        """Test document.field == 'literal' parsing."""
        from ragguard.filters.backends.pinecone import _build_pinecone_condition_filter

        filter_dict = _build_pinecone_condition_filter(
            "document.status == 'active'",
            {"id": "alice"}
        )
        assert filter_dict is not None
        assert "status" in filter_dict

    def test_parse_user_in_document(self):
        """Test user.field in document.array_field parsing."""
        from ragguard.filters.backends.pinecone import _build_pinecone_condition_filter

        filter_dict = _build_pinecone_condition_filter(
            "user.role in document.allowed_roles",
            {"id": "alice", "role": "admin"}
        )
        assert filter_dict is not None
        # Pinecone uses $in for membership
        assert "allowed_roles" in str(filter_dict)


class TestPineconeCompiledExpressions:
    """Tests for Pinecone compiled expression handling."""

    def test_compiled_expression_or(self):
        """Test compiled OR expression."""
        from ragguard.filters.backends.pinecone import _build_pinecone_from_compiled_node
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledExpression,
            CompiledValue,
            ConditionOperator,
            LogicalOperator,
            ValueType,
        )

        cond1 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="active", field_path=()),
            original="test"
        )
        cond2 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="pending", field_path=()),
            original="test"
        )
        expr = CompiledExpression(operator=LogicalOperator.OR, children=[cond1, cond2], original="test")

        result = _build_pinecone_from_compiled_node(expr, {"id": "alice"})

        assert result is not None
        assert "$or" in result

    def test_compiled_expression_and(self):
        """Test compiled AND expression."""
        from ragguard.filters.backends.pinecone import _build_pinecone_from_compiled_node
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledExpression,
            CompiledValue,
            ConditionOperator,
            LogicalOperator,
            ValueType,
        )

        cond1 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("public",)),
            right=CompiledValue(value_type=ValueType.LITERAL_BOOL, value=True, field_path=()),
            original="test"
        )
        cond2 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="active", field_path=()),
            original="test"
        )
        expr = CompiledExpression(operator=LogicalOperator.AND, children=[cond1, cond2], original="test")

        result = _build_pinecone_from_compiled_node(expr, {"id": "alice"})

        assert result is not None
        assert "$and" in result


# ============================================================
# Weaviate String Parsing Tests
# ============================================================

class TestWeaviateStringParsing:
    """Tests for Weaviate string parsing fallback paths."""

    def test_parse_user_equals_document(self):
        """Test user.field == document.field parsing."""
        from ragguard.filters.backends.weaviate import _build_weaviate_condition_filter

        filter_dict = _build_weaviate_condition_filter(
            "user.department == document.department",
            {"id": "alice", "department": "engineering"}
        )
        assert filter_dict is not None
        assert "path" in filter_dict

    def test_parse_user_equals_document_null(self):
        """Test user.field == document.field with null user value."""
        from ragguard.filters.backends.weaviate import _build_weaviate_condition_filter

        filter_dict = _build_weaviate_condition_filter(
            "user.department == document.department",
            {"id": "alice"}  # No department
        )
        # Should return deny-all filter
        assert filter_dict is not None
        assert "__ragguard_deny_all__" in str(filter_dict)

    def test_parse_document_equals_literal(self):
        """Test document.field == 'literal' parsing."""
        from ragguard.filters.backends.weaviate import _build_weaviate_condition_filter

        filter_dict = _build_weaviate_condition_filter(
            "document.status == 'active'",
            {"id": "alice"}
        )
        assert filter_dict is not None
        assert "status" in str(filter_dict)

    def test_parse_user_in_document(self):
        """Test user.field in document.array_field parsing."""
        from ragguard.filters.backends.weaviate import _build_weaviate_condition_filter

        filter_dict = _build_weaviate_condition_filter(
            "user.role in document.allowed_roles",
            {"id": "alice", "role": "admin"}
        )
        assert filter_dict is not None


class TestWeaviateCompiledExpressions:
    """Tests for Weaviate compiled expression handling."""

    def test_compiled_expression_or(self):
        """Test compiled OR expression."""
        from ragguard.filters.backends.weaviate import _build_weaviate_from_compiled_node
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledExpression,
            CompiledValue,
            ConditionOperator,
            LogicalOperator,
            ValueType,
        )

        cond1 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="active", field_path=()),
            original="test"
        )
        cond2 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="pending", field_path=()),
            original="test"
        )
        expr = CompiledExpression(operator=LogicalOperator.OR, children=[cond1, cond2], original="test")

        result = _build_weaviate_from_compiled_node(expr, {"id": "alice"})

        assert result is not None
        assert "operator" in result
        assert result["operator"] == "Or"

    def test_compiled_expression_and(self):
        """Test compiled AND expression."""
        from ragguard.filters.backends.weaviate import _build_weaviate_from_compiled_node
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledExpression,
            CompiledValue,
            ConditionOperator,
            LogicalOperator,
            ValueType,
        )

        cond1 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("public",)),
            right=CompiledValue(value_type=ValueType.LITERAL_BOOL, value=True, field_path=()),
            original="test"
        )
        cond2 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="active", field_path=()),
            original="test"
        )
        expr = CompiledExpression(operator=LogicalOperator.AND, children=[cond1, cond2], original="test")

        result = _build_weaviate_from_compiled_node(expr, {"id": "alice"})

        assert result is not None
        assert "operator" in result
        assert result["operator"] == "And"
