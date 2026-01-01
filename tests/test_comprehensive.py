"""
Comprehensive tests to improve coverage to 95%+.

Tests for:
- Filter backends (pgvector, weaviate, milvus, pinecone, qdrant, elasticsearch, chromadb)
- Exceptions module
- Policy engine edge cases
- Other low-coverage modules
"""

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# ============================================================================
# Exceptions Module Tests
# ============================================================================

class TestRAGGuardExceptions:
    """Tests for ragguard/exceptions.py"""

    def test_ragguard_error_base(self):
        """Test RAGGuardError base exception."""
        from ragguard.exceptions import RAGGuardError

        # With message
        err = RAGGuardError("test message")
        assert str(err) == "test message"
        assert err.message == "test message"

        # Without message
        err2 = RAGGuardError()
        assert err2.message == ""

    def test_policy_errors(self):
        """Test policy-related exceptions."""
        from ragguard.exceptions import PolicyError, PolicyParseError, PolicyValidationError

        err = PolicyError("policy error")
        assert "policy error" in str(err)

        parse_err = PolicyParseError("parse failed")
        assert "parse failed" in str(parse_err)

        val_err = PolicyValidationError("validation failed")
        assert "validation failed" in str(val_err)

    def test_policy_evaluation_error(self):
        """Test PolicyEvaluationError."""
        from ragguard.exceptions import PolicyEvaluationError

        err = PolicyEvaluationError("evaluation failed")
        assert "evaluation failed" in str(err)

    def test_filter_build_error(self):
        """Test FilterBuildError."""
        from ragguard.exceptions import FilterBuildError

        err = FilterBuildError("filter build failed")
        assert "filter build failed" in str(err)

    def test_unsupported_condition_error(self):
        """Test UnsupportedConditionError."""
        from ragguard.exceptions import UnsupportedConditionError

        # Without reason
        err = UnsupportedConditionError("user.x == doc.y", "qdrant")
        assert "user.x == doc.y" in str(err)
        assert "qdrant" in str(err)

        # With reason
        err2 = UnsupportedConditionError("complex_cond", "pinecone", "not supported")
        assert "complex_cond" in str(err2)
        assert "not supported" in str(err2)

    def test_retriever_connection_error(self):
        """Test RetrieverConnectionError."""
        from ragguard.exceptions import RetrieverConnectionError

        # Without message or cause
        err = RetrieverConnectionError("qdrant")
        assert "qdrant" in str(err)
        assert err.retryable is True

        # With message
        err2 = RetrieverConnectionError("qdrant", "server unreachable")
        assert "server unreachable" in str(err2)

        # With cause
        cause = ConnectionError("network down")
        err3 = RetrieverConnectionError("qdrant", "failed", cause)
        assert "ConnectionError" in str(err3)
        assert "network down" in str(err3)

    def test_retriever_timeout_error(self):
        """Test RetrieverTimeoutError."""
        from ragguard.exceptions import RetrieverTimeoutError

        # Without timeout
        err = RetrieverTimeoutError("pinecone", "search")
        assert "pinecone" in str(err)
        assert "search" in str(err)
        assert err.retryable is True

        # With timeout
        err2 = RetrieverTimeoutError("pinecone", "query", 30.0)
        assert "30" in str(err2)

    def test_health_check_error(self):
        """Test HealthCheckError."""
        from ragguard.exceptions import HealthCheckError

        # Basic
        err = HealthCheckError("milvus")
        assert "milvus" in str(err)

        # With message
        err2 = HealthCheckError("milvus", "connection refused")
        assert "connection refused" in str(err2)

        # With cause
        cause = Exception("internal error")
        err3 = HealthCheckError("milvus", "", cause)
        assert "internal error" in str(err3)

    def test_backend_error(self):
        """Test BackendError."""
        from ragguard.exceptions import BackendError

        # Basic
        err = BackendError("weaviate")
        assert "weaviate" in str(err)

        # With message
        err2 = BackendError("weaviate", "schema error")
        assert "schema error" in str(err2)

        # With cause
        cause = ValueError("invalid")
        err3 = BackendError("weaviate", "", cause)
        assert "ValueError" in str(err3)

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        from ragguard.exceptions import RateLimitError

        # Without retry_after
        err = RateLimitError("openai")
        assert "openai" in str(err)
        assert err.retryable is True

        # With retry_after
        err2 = RateLimitError("openai", 60.0)
        assert "60" in str(err2)

    def test_quota_exceeded_error(self):
        """Test QuotaExceededError."""
        from ragguard.exceptions import QuotaExceededError

        # Without message
        err = QuotaExceededError("pinecone")
        assert "pinecone" in str(err)
        assert err.retryable is False

        # With message
        err2 = QuotaExceededError("pinecone", "monthly limit reached")
        assert "monthly limit" in str(err2)

    def test_retriever_permission_error(self):
        """Test RetrieverPermissionError."""
        from ragguard.exceptions import RetrieverPermissionError

        # Basic
        err = RetrieverPermissionError("qdrant")
        assert "qdrant" in str(err)
        assert err.retryable is False

        # With resource
        err2 = RetrieverPermissionError("qdrant", "collection_xyz")
        assert "collection_xyz" in str(err2)

        # With message
        err3 = RetrieverPermissionError("qdrant", "", "API key invalid")
        assert "API key" in str(err3)

    def test_configuration_error(self):
        """Test ConfigurationError."""
        from ragguard.exceptions import ConfigurationError

        # With parameter only
        err = ConfigurationError(parameter="timeout")
        assert "timeout" in str(err)
        assert err.retryable is False

        # With message and parameter
        err2 = ConfigurationError("must be positive", "timeout")
        assert "must be positive" in str(err2)
        assert "timeout" in str(err2)

        # With message only
        err3 = ConfigurationError("general config error")
        assert "general config error" in str(err3)

    def test_retryable_exceptions_set(self):
        """Test RETRYABLE_EXCEPTIONS tuple."""
        from ragguard.exceptions import (
            RETRYABLE_EXCEPTIONS,
            RateLimitError,
            RetrieverConnectionError,
            RetrieverTimeoutError,
        )

        assert RetrieverConnectionError in RETRYABLE_EXCEPTIONS
        assert RetrieverTimeoutError in RETRYABLE_EXCEPTIONS
        assert RateLimitError in RETRYABLE_EXCEPTIONS

    def test_non_retryable_exceptions_set(self):
        """Test NON_RETRYABLE_EXCEPTIONS tuple."""
        from ragguard.exceptions import (
            NON_RETRYABLE_EXCEPTIONS,
            ConfigurationError,
            PolicyError,
            QuotaExceededError,
            RetrieverPermissionError,
        )

        assert QuotaExceededError in NON_RETRYABLE_EXCEPTIONS
        assert RetrieverPermissionError in NON_RETRYABLE_EXCEPTIONS
        assert ConfigurationError in NON_RETRYABLE_EXCEPTIONS
        assert PolicyError in NON_RETRYABLE_EXCEPTIONS


# ============================================================================
# pgvector Filter Backend Tests
# ============================================================================

class TestPgvectorFilterBackend:
    """Extended tests for ragguard/filters/backends/pgvector.py"""

    def test_to_pgvector_filter_no_matching_rules(self):
        """Test when no rules match the user."""
        from ragguard import Policy
        from ragguard.filters.backends.pgvector import to_pgvector_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "admin-only", "allow": {"roles": ["admin"]}}],
            "default": "deny"
        })

        # User without admin role
        clause, params = to_pgvector_filter(policy, {"id": "alice", "roles": ["user"]})
        assert clause == "WHERE FALSE"
        assert params == []

    def test_to_pgvector_filter_default_allow(self):
        """Test default allow when no rules match."""
        from ragguard import Policy
        from ragguard.filters.backends.pgvector import to_pgvector_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "admin-only", "allow": {"roles": ["admin"]}}],
            "default": "allow"
        })

        clause, params = to_pgvector_filter(policy, {"id": "alice", "roles": ["user"]})
        assert clause == ""
        assert params == []

    def test_to_pgvector_filter_with_match(self):
        """Test filter with match conditions."""
        from ragguard import Policy
        from ragguard.filters.backends.pgvector import to_pgvector_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "status-check",
                "match": {"status": "active"},
                "allow": {"roles": ["user"]}
            }],
            "default": "deny"
        })

        clause, params = to_pgvector_filter(policy, {"id": "alice", "roles": ["user"]})
        assert "status = %s" in clause
        assert "active" in params

    def test_to_pgvector_filter_with_match_list(self):
        """Test filter with match list conditions."""
        from ragguard import Policy
        from ragguard.filters.backends.pgvector import to_pgvector_filter

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "type-check",
                "match": {"type": ["doc", "report", "memo"]},
                "allow": {"roles": ["user"]}
            }],
            "default": "deny"
        })

        clause, params = to_pgvector_filter(policy, {"id": "alice", "roles": ["user"]})
        assert "IN" in clause
        assert "doc" in params

    def test_pgvector_legacy_exists_condition(self):
        """Test legacy exists condition parsing."""
        from ragguard.filters.backends.pgvector import _parse_pgvector_legacy_condition

        # exists
        clause, params = _parse_pgvector_legacy_condition("document.author exists", {})
        assert "IS NOT NULL" in clause
        assert "author" in clause

        # not exists
        clause, params = _parse_pgvector_legacy_condition("document.draft not exists", {})
        assert "IS NULL" in clause
        assert "draft" in clause

    def test_pgvector_legacy_not_equals(self):
        """Test legacy != condition parsing."""
        from ragguard.filters.backends.pgvector import _parse_pgvector_legacy_condition

        clause, params = _parse_pgvector_legacy_condition("document.status != 'deleted'", {})
        assert "!=" in clause
        assert "deleted" in params

    def test_pgvector_legacy_not_in(self):
        """Test legacy not in condition parsing."""
        from ragguard.filters.backends.pgvector import _parse_pgvector_legacy_condition

        clause, params = _parse_pgvector_legacy_condition(
            "user.id not in document.blocked_users",
            {"id": "alice"}
        )
        assert "NOT" in clause
        assert "alice" in params

    def test_pgvector_legacy_not_in_null_user_value(self):
        """Test legacy not in with null user value."""
        from ragguard.filters.backends.pgvector import _parse_pgvector_legacy_condition

        clause, params = _parse_pgvector_legacy_condition(
            "user.id not in document.blocked_users",
            {}  # No id
        )
        assert "1 = 1" in clause  # NULL not in list is always true

    def test_pgvector_legacy_in_condition(self):
        """Test legacy in condition parsing."""
        from ragguard.filters.backends.pgvector import _parse_pgvector_legacy_condition

        clause, params = _parse_pgvector_legacy_condition(
            "user.department in document.allowed_depts",
            {"department": "engineering"}
        )
        assert "ANY" in clause
        assert "engineering" in params

    def test_pgvector_compiled_greater_than(self):
        """Test compiled GREATER_THAN operator."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        cond = CompiledCondition(
            operator=ConditionOperator.GREATER_THAN,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("level",)),
            right=CompiledValue(value_type=ValueType.LITERAL_NUMBER, value=5, field_path=()),
            original="test"
        )
        clause, params = _build_pgvector_from_condition(cond, {})
        assert ">" in clause
        assert 5 in params

    def test_pgvector_compiled_not_equals(self):
        """Test compiled NOT_EQUALS operator."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        cond = CompiledCondition(
            operator=ConditionOperator.NOT_EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="deleted", field_path=()),
            original="test"
        )

        clause, params = _build_pgvector_from_condition(cond, {})
        assert "!=" in clause
        assert "deleted" in params

    def test_pgvector_compiled_not_in_with_list(self):
        """Test compiled NOT_IN with list."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        cond = CompiledCondition(
            operator=ConditionOperator.NOT_IN,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_LIST, value=["draft", "deleted"], field_path=()),
            original="test"
        )

        clause, params = _build_pgvector_from_condition(cond, {})
        assert "NOT IN" in clause
        assert "draft" in params

    def test_pgvector_compiled_not_in_empty_list(self):
        """Test compiled NOT_IN with empty list."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        cond = CompiledCondition(
            operator=ConditionOperator.NOT_IN,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_LIST, value=[], field_path=()),
            original="test"
        )

        clause, params = _build_pgvector_from_condition(cond, {})
        assert "TRUE" in clause  # NOT IN empty list is always true

    def test_pgvector_compiled_not_in_user_null(self):
        """Test compiled NOT_IN with null user value."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        cond = CompiledCondition(
            operator=ConditionOperator.NOT_IN,
            left=CompiledValue(value_type=ValueType.USER_FIELD, value=None, field_path=("id",)),
            right=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("blocked",)),
            original="test"
        )

        clause, params = _build_pgvector_from_condition(cond, {})  # No id
        assert "TRUE" in clause

    def test_pgvector_compiled_comparison_operators(self):
        """Test compiled comparison operators."""
        from ragguard.filters.backends.pgvector import _build_pgvector_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        # Greater than
        cond = CompiledCondition(
            operator=ConditionOperator.GREATER_THAN,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("level",)),
            right=CompiledValue(value_type=ValueType.LITERAL_NUMBER, value=5, field_path=()),
            original="test"
        )
        clause, params = _build_pgvector_from_condition(cond, {})
        assert ">" in clause
        assert 5 in params

        # Less than
        cond.operator = ConditionOperator.LESS_THAN
        clause, params = _build_pgvector_from_condition(cond, {})
        assert "<" in clause

        # Greater than or equal
        cond.operator = ConditionOperator.GREATER_THAN_OR_EQUAL
        clause, params = _build_pgvector_from_condition(cond, {})
        assert ">=" in clause

        # Less than or equal
        cond.operator = ConditionOperator.LESS_THAN_OR_EQUAL
        clause, params = _build_pgvector_from_condition(cond, {})
        assert "<=" in clause


# ============================================================================
# Weaviate Filter Backend Tests
# ============================================================================

class TestWeaviateFilterBackend:
    """Extended tests for ragguard/filters/backends/weaviate.py"""

    def test_weaviate_compiled_not_equals(self):
        """Test compiled NOT_EQUALS operator."""
        from ragguard.filters.backends.weaviate import _build_weaviate_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        cond = CompiledCondition(
            operator=ConditionOperator.NOT_EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="deleted", field_path=()),
            original="test"
        )

        result = _build_weaviate_from_condition(cond, {})
        assert result is not None

    def test_weaviate_compiled_in_with_list(self):
        """Test compiled IN with non-empty list."""
        from ragguard.filters.backends.weaviate import _build_weaviate_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        cond = CompiledCondition(
            operator=ConditionOperator.IN,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_LIST, value=["active", "pending"], field_path=()),
            original="test"
        )

        result = _build_weaviate_from_condition(cond, {})
        assert result is not None

    def test_weaviate_compiled_comparison_operators(self):
        """Test compiled comparison operators."""
        from ragguard.filters.backends.weaviate import _build_weaviate_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        for op in [ConditionOperator.GREATER_THAN, ConditionOperator.LESS_THAN,
                   ConditionOperator.GREATER_THAN_OR_EQUAL, ConditionOperator.LESS_THAN_OR_EQUAL]:
            cond = CompiledCondition(
                operator=op,
                left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("level",)),
                right=CompiledValue(value_type=ValueType.LITERAL_NUMBER, value=5, field_path=()),
                original="test"
            )
            result = _build_weaviate_from_condition(cond, {})
            assert result is not None

    def test_weaviate_compiled_not_in(self):
        """Test compiled NOT_IN operator."""
        from ragguard.filters.backends.weaviate import _build_weaviate_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        cond = CompiledCondition(
            operator=ConditionOperator.NOT_IN,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_LIST, value=["draft", "deleted"], field_path=()),
            original="test"
        )

        result = _build_weaviate_from_condition(cond, {})
        assert result is not None


# ============================================================================
# Milvus Filter Backend Tests
# ============================================================================

class TestMilvusFilterBackend:
    """Extended tests for ragguard/filters/backends/milvus.py"""

    def test_milvus_compiled_not_equals(self):
        """Test compiled NOT_EQUALS operator."""
        from ragguard.filters.backends.milvus import _build_milvus_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        cond = CompiledCondition(
            operator=ConditionOperator.NOT_EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="deleted", field_path=()),
            original="test"
        )

        result = _build_milvus_from_condition(cond, {})
        assert result is not None and "!=" in result

    def test_milvus_compiled_in_with_list(self):
        """Test compiled IN with non-empty list."""
        from ragguard.filters.backends.milvus import _build_milvus_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        cond = CompiledCondition(
            operator=ConditionOperator.IN,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_LIST, value=["active", "pending"], field_path=()),
            original="test"
        )

        result = _build_milvus_from_condition(cond, {})
        assert result is not None and "in" in result.lower()

    def test_milvus_compiled_comparison_operators(self):
        """Test compiled comparison operators."""
        from ragguard.filters.backends.milvus import _build_milvus_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        for op, symbol in [(ConditionOperator.GREATER_THAN, ">"),
                           (ConditionOperator.LESS_THAN, "<"),
                           (ConditionOperator.GREATER_THAN_OR_EQUAL, ">="),
                           (ConditionOperator.LESS_THAN_OR_EQUAL, "<=")]:
            cond = CompiledCondition(
                operator=op,
                left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("level",)),
                right=CompiledValue(value_type=ValueType.LITERAL_NUMBER, value=5, field_path=()),
                original="test"
            )
            result = _build_milvus_from_condition(cond, {})
            assert result is not None and symbol in result


# ============================================================================
# Pinecone Filter Backend Tests
# ============================================================================

class TestPineconeFilterBackend:
    """Extended tests for ragguard/filters/backends/pinecone.py"""

    def test_pinecone_compiled_not_equals(self):
        """Test compiled NOT_EQUALS operator."""
        from ragguard.filters.backends.pinecone import _build_pinecone_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        cond = CompiledCondition(
            operator=ConditionOperator.NOT_EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="deleted", field_path=()),
            original="test"
        )

        result = _build_pinecone_from_condition(cond, {})
        assert result is not None and "$ne" in str(result)

    def test_pinecone_compiled_not_in(self):
        """Test compiled NOT_IN operator."""
        from ragguard.filters.backends.pinecone import _build_pinecone_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        cond = CompiledCondition(
            operator=ConditionOperator.NOT_IN,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_LIST, value=["draft", "deleted"], field_path=()),
            original="test"
        )

        result = _build_pinecone_from_condition(cond, {})
        assert result is not None and "$nin" in str(result)

    def test_pinecone_compiled_comparison_operators(self):
        """Test compiled comparison operators."""
        from ragguard.filters.backends.pinecone import _build_pinecone_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        for op, key in [(ConditionOperator.GREATER_THAN, "$gt"),
                        (ConditionOperator.LESS_THAN, "$lt"),
                        (ConditionOperator.GREATER_THAN_OR_EQUAL, "$gte"),
                        (ConditionOperator.LESS_THAN_OR_EQUAL, "$lte")]:
            cond = CompiledCondition(
                operator=op,
                left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("level",)),
                right=CompiledValue(value_type=ValueType.LITERAL_NUMBER, value=5, field_path=()),
                original="test"
            )
            result = _build_pinecone_from_condition(cond, {})
            assert result is not None and key in str(result)


# ============================================================================
# Qdrant Filter Backend Tests
# ============================================================================

class TestQdrantFilterBackend:
    """Extended tests for ragguard/filters/backends/qdrant.py"""

    def _get_qdrant_models(self):
        """Get Qdrant models for testing."""
        try:
            from qdrant_client import models
            return models
        except ImportError:
            pytest.skip("qdrant-client not installed")

    def test_qdrant_compiled_not_equals(self):
        """Test compiled NOT_EQUALS operator."""
        models = self._get_qdrant_models()
        from ragguard.filters.backends.qdrant import _build_qdrant_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        cond = CompiledCondition(
            operator=ConditionOperator.NOT_EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="deleted", field_path=()),
            original="test"
        )

        result = _build_qdrant_from_condition(cond, {}, models)
        assert result is not None

    def test_qdrant_compiled_not_in(self):
        """Test compiled NOT_IN operator."""
        models = self._get_qdrant_models()
        from ragguard.filters.backends.qdrant import _build_qdrant_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        cond = CompiledCondition(
            operator=ConditionOperator.NOT_IN,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_LIST, value=["draft", "deleted"], field_path=()),
            original="test"
        )

        result = _build_qdrant_from_condition(cond, {}, models)
        assert result is not None

    def test_qdrant_compiled_comparison_operators(self):
        """Test compiled comparison operators."""
        models = self._get_qdrant_models()
        from ragguard.filters.backends.qdrant import _build_qdrant_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        for op in [ConditionOperator.GREATER_THAN, ConditionOperator.LESS_THAN,
                   ConditionOperator.GREATER_THAN_OR_EQUAL, ConditionOperator.LESS_THAN_OR_EQUAL]:
            cond = CompiledCondition(
                operator=op,
                left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("level",)),
                right=CompiledValue(value_type=ValueType.LITERAL_NUMBER, value=5, field_path=()),
                original="test"
            )
            result = _build_qdrant_from_condition(cond, {}, models)
            assert result is not None


# ============================================================================
# Elasticsearch Filter Backend Tests
# ============================================================================

class TestElasticsearchFilterBackend:
    """Extended tests for ragguard/filters/backends/elasticsearch.py"""

    def test_elasticsearch_compiled_not_equals(self):
        """Test compiled NOT_EQUALS operator."""
        from ragguard.filters.backends.elasticsearch import _build_elasticsearch_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        cond = CompiledCondition(
            operator=ConditionOperator.NOT_EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="deleted", field_path=()),
            original="test"
        )

        result = _build_elasticsearch_from_condition(cond, {})
        assert result is not None

    def test_elasticsearch_compiled_comparison_operators(self):
        """Test compiled comparison operators."""
        from ragguard.filters.backends.elasticsearch import _build_elasticsearch_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        for op in [ConditionOperator.GREATER_THAN, ConditionOperator.LESS_THAN,
                   ConditionOperator.GREATER_THAN_OR_EQUAL, ConditionOperator.LESS_THAN_OR_EQUAL]:
            cond = CompiledCondition(
                operator=op,
                left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("level",)),
                right=CompiledValue(value_type=ValueType.LITERAL_NUMBER, value=5, field_path=()),
                original="test"
            )
            result = _build_elasticsearch_from_condition(cond, {})
            assert result is not None


# ============================================================================
# ChromaDB Filter Backend Tests
# ============================================================================

class TestChromaDBFilterBackend:
    """Extended tests for ragguard/filters/backends/chromadb.py"""

    def test_chromadb_compiled_not_equals(self):
        """Test compiled NOT_EQUALS operator."""
        from ragguard.filters.backends.chromadb import _build_chromadb_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        cond = CompiledCondition(
            operator=ConditionOperator.NOT_EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="deleted", field_path=()),
            original="test"
        )

        result = _build_chromadb_from_condition(cond, {})
        assert result is not None and "$ne" in str(result)

    def test_chromadb_compiled_not_in(self):
        """Test compiled NOT_IN operator."""
        from ragguard.filters.backends.chromadb import _build_chromadb_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        cond = CompiledCondition(
            operator=ConditionOperator.NOT_IN,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_LIST, value=["draft", "deleted"], field_path=()),
            original="test"
        )

        result = _build_chromadb_from_condition(cond, {})
        assert result is not None and "$nin" in str(result)

    def test_chromadb_compiled_comparison_operators(self):
        """Test compiled comparison operators."""
        from ragguard.filters.backends.chromadb import _build_chromadb_from_condition
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledValue,
            ConditionOperator,
            ValueType,
        )

        for op, key in [(ConditionOperator.GREATER_THAN, "$gt"),
                        (ConditionOperator.LESS_THAN, "$lt"),
                        (ConditionOperator.GREATER_THAN_OR_EQUAL, "$gte"),
                        (ConditionOperator.LESS_THAN_OR_EQUAL, "$lte")]:
            cond = CompiledCondition(
                operator=op,
                left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("level",)),
                right=CompiledValue(value_type=ValueType.LITERAL_NUMBER, value=5, field_path=()),
                original="test"
            )
            result = _build_chromadb_from_condition(cond, {})
            assert result is not None and key in str(result)


# ============================================================================
# Integration/__init__.py Tests
# ============================================================================

class TestIntegrationsInit:
    """Tests for ragguard/integrations/__init__.py"""

    def test_langchain_import_available(self):
        """Test LangChain integration import when available."""
        try:
            from ragguard.integrations import PermissionedRetriever
            assert PermissionedRetriever is not None
        except ImportError:
            pytest.skip("LangChain not installed")

    def test_llama_index_import_available(self):
        """Test LlamaIndex integration import when available."""
        try:
            from ragguard.integrations import RAGGuardRetriever
            assert RAGGuardRetriever is not None
        except ImportError:
            pytest.skip("LlamaIndex not installed")

    def test_langgraph_import_available(self):
        """Test LangGraph integration import when available."""
        try:
            from ragguard.integrations import create_ragguard_node
            assert create_ragguard_node is not None
        except ImportError:
            pytest.skip("LangGraph not installed")

    def test_bedrock_import_available(self):
        """Test AWS Bedrock integration import when available."""
        try:
            from ragguard.integrations import BedrockKnowledgeBaseRetriever
            assert BedrockKnowledgeBaseRetriever is not None
        except ImportError:
            pytest.skip("Boto3 not installed")

    def test_mcp_import_available(self):
        """Test MCP integration import when available."""
        try:
            from ragguard.integrations import create_ragguard_mcp_server
            assert create_ragguard_mcp_server is not None
        except ImportError:
            pytest.skip("MCP not installed")
