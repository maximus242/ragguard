"""
Tests for filter validation and FilterResult semantics.

These tests ensure that filter builders return unambiguous results
and that the policy engine correctly detects deny-all patterns.
"""

import logging

import pytest

from ragguard.policy import Policy
from ragguard.policy.engine import PolicyEngine, _is_deny_all_filter, _validate_filter_result
from ragguard.types import FilterResult, FilterResultType


class TestDenyAllDetection:
    """Test detection of deny-all filter patterns."""

    def test_detect_pgvector_deny_all(self):
        """Test detection of pgvector WHERE FALSE pattern."""
        assert _is_deny_all_filter(("WHERE FALSE", []), "pgvector") is True
        assert _is_deny_all_filter(("WHERE 1=0", []), "pgvector") is True
        assert _is_deny_all_filter(("WHERE status = %s", ["active"]), "pgvector") is False

    def test_detect_milvus_deny_all(self):
        """Test detection of milvus 1==0 pattern."""
        assert _is_deny_all_filter("1 == 0", "milvus") is True
        assert _is_deny_all_filter("1==0", "milvus") is True
        assert _is_deny_all_filter("status == 'active'", "milvus") is False

    def test_detect_chromadb_deny_all(self):
        """Test detection of ChromaDB deny pattern."""
        deny_filter = {"__deny_all__": {"$eq": "__never_match__"}}
        assert _is_deny_all_filter(deny_filter, "chromadb") is True

        nested_deny = {"$and": [{"__deny_all__": {"$eq": "__never_match__"}}]}
        assert _is_deny_all_filter(nested_deny, "chromadb") is True

        normal_filter = {"department": {"$eq": "engineering"}}
        assert _is_deny_all_filter(normal_filter, "chromadb") is False

    def test_detect_elasticsearch_deny_all(self):
        """Test detection of Elasticsearch deny pattern."""
        deny_filter = {
            "bool": {
                "must": [{"term": {"__deny_all__": "__never_match__"}}]
            }
        }
        assert _is_deny_all_filter(deny_filter, "elasticsearch") is True

        normal_filter = {"bool": {"must": [{"term": {"status": "active"}}]}}
        assert _is_deny_all_filter(normal_filter, "elasticsearch") is False

    def test_none_is_not_deny_all(self):
        """Test that None is not considered deny-all."""
        assert _is_deny_all_filter(None, "chromadb") is False
        assert _is_deny_all_filter(None, "qdrant") is False

    def test_empty_dict_is_not_deny_all(self):
        """Test that empty dict is not considered deny-all (but may be ambiguous)."""
        assert _is_deny_all_filter({}, "chromadb") is False


class TestFilterValidation:
    """Test filter result validation."""

    def test_none_with_deny_default_logs_debug(self, caplog):
        """Test that None filter with deny default logs debug message."""
        with caplog.at_level(logging.DEBUG):
            filter_obj, warning = _validate_filter_result(
                None, "chromadb", {"id": "alice"}, "deny"
            )
            assert filter_obj is None
            assert warning is None
            assert "allow all" in caplog.text.lower() or len(caplog.records) == 1

    def test_empty_dict_warns(self, caplog):
        """Test that empty dict filter logs warning."""
        with caplog.at_level(logging.WARNING):
            filter_obj, warning = _validate_filter_result(
                {}, "chromadb", {"id": "alice"}, "deny"
            )
            assert filter_obj == {}
            assert warning is not None
            assert "empty dict" in warning.lower()

    def test_empty_list_warns(self, caplog):
        """Test that empty list filter logs warning."""
        with caplog.at_level(logging.WARNING):
            filter_obj, warning = _validate_filter_result(
                [], "chromadb", {"id": "alice"}, "deny"
            )
            assert filter_obj == []
            assert warning is not None
            assert "empty list" in warning.lower()

    def test_valid_filter_no_warning(self, caplog):
        """Test that valid filter produces no warning."""
        with caplog.at_level(logging.WARNING):
            filter_obj, warning = _validate_filter_result(
                {"status": {"$eq": "active"}}, "chromadb", {"id": "alice"}, "deny"
            )
            assert warning is None


class TestFilterResult:
    """Test FilterResult class methods."""

    def test_allow_all(self):
        """Test FilterResult.allow_all() factory."""
        result = FilterResult.allow_all()
        assert result.result_type == FilterResultType.ALLOW_ALL
        assert result.filter is None
        assert result.is_allow_all is True
        assert result.is_deny_all is False
        assert result.is_conditional is False

    def test_deny_all(self):
        """Test FilterResult.deny_all() factory."""
        result = FilterResult.deny_all(reason="No access")
        assert result.result_type == FilterResultType.DENY_ALL
        assert result.filter is None
        assert result.reason == "No access"
        assert result.is_deny_all is True
        assert result.is_allow_all is False
        assert result.is_conditional is False

    def test_conditional(self):
        """Test FilterResult.conditional() factory."""
        filter_obj = {"status": {"$eq": "active"}}
        result = FilterResult.conditional(filter_obj)
        assert result.result_type == FilterResultType.CONDITIONAL
        assert result.filter == filter_obj
        assert result.is_conditional is True
        assert result.is_allow_all is False
        assert result.is_deny_all is False


class TestPolicyEngineFilterResult:
    """Test PolicyEngine.to_filter_result() method."""

    @pytest.fixture
    def allow_all_policy(self):
        """Create a policy that allows everyone."""
        return Policy.from_dict({
            "version": "1",
            "rules": [{"name": "allow-all", "allow": {"everyone": True}}],
            "default": "deny"
        })

    @pytest.fixture
    def deny_all_policy(self):
        """Create a policy that denies everyone except admins."""
        return Policy.from_dict({
            "version": "1",
            "rules": [{"name": "admin-only", "allow": {"roles": ["admin"]}}],
            "default": "deny"
        })

    @pytest.fixture
    def conditional_policy(self):
        """Create a policy with conditional access."""
        return Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "dept-access",
                "allow": {"conditions": ["user.department == document.department"]}
            }],
            "default": "deny"
        })

    def test_to_filter_result_allow_all(self, allow_all_policy):
        """Test that allow-all policy returns ALLOW_ALL result."""
        engine = PolicyEngine(allow_all_policy)
        user = {"id": "alice", "roles": ["user"]}

        result = engine.to_filter_result(user, "chromadb")
        assert result.is_allow_all is True

    def test_to_filter_result_deny_all(self, deny_all_policy):
        """Test that deny-all scenario returns DENY_ALL result."""
        engine = PolicyEngine(deny_all_policy)
        user = {"id": "bob", "roles": ["user"]}  # Not an admin

        result = engine.to_filter_result(user, "chromadb")
        assert result.is_deny_all is True

    def test_to_filter_result_conditional(self, conditional_policy):
        """Test that conditional policy returns CONDITIONAL result."""
        engine = PolicyEngine(conditional_policy)
        user = {"id": "alice", "department": "engineering"}

        result = engine.to_filter_result(user, "chromadb")
        assert result.is_conditional is True
        assert result.filter is not None

    def test_is_deny_all_filter_method(self, deny_all_policy):
        """Test PolicyEngine.is_deny_all_filter() method."""
        engine = PolicyEngine(deny_all_policy)
        user = {"id": "bob", "roles": ["user"]}

        filter_obj = engine.to_filter(user, "chromadb")
        assert engine.is_deny_all_filter(filter_obj, "chromadb") is True

    def test_is_deny_all_filter_with_normal_filter(self, conditional_policy):
        """Test that normal filter is not detected as deny-all."""
        engine = PolicyEngine(conditional_policy)
        user = {"id": "alice", "department": "engineering"}

        filter_obj = engine.to_filter(user, "chromadb")
        assert engine.is_deny_all_filter(filter_obj, "chromadb") is False


class TestFilterResultUsagePattern:
    """Test usage patterns for FilterResult in retrievers."""

    def test_skip_query_on_deny_all(self):
        """Test that retrievers can skip queries when deny-all is detected."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "admin-only", "allow": {"roles": ["admin"]}}],
            "default": "deny"
        })
        engine = PolicyEngine(policy)
        user = {"id": "bob", "roles": ["user"]}

        result = engine.to_filter_result(user, "chromadb")

        # Retriever pattern: skip query entirely if deny-all
        if result.is_deny_all:
            # Would skip database query and return empty list
            results = []
        else:
            # Would execute query with filter
            results = ["doc1", "doc2"]  # Mock results

        assert results == []

    def test_no_filter_on_allow_all(self):
        """Test that retrievers can skip filtering when allow-all is detected."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "allow-all", "allow": {"everyone": True}}],
            "default": "deny"
        })
        engine = PolicyEngine(policy)
        user = {"id": "alice"}

        result = engine.to_filter_result(user, "chromadb")

        # Retriever pattern: no filter needed for allow-all
        if result.is_allow_all:
            filter_to_use = None  # No filter
        elif result.is_deny_all:
            filter_to_use = "SKIP"  # Skip query
        else:
            filter_to_use = result.filter

        assert filter_to_use is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
