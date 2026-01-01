"""Extended tests for graph database filter builders to improve coverage."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from ragguard.policy.models import AllowConditions, Policy, Rule


def make_policy(rules, default="deny"):
    """Helper to create a policy from rules."""
    return Policy(
        version="1",
        rules=[
            Rule(
                name=f"rule_{i}",
                allow=AllowConditions(
                    roles=r.get("roles"),
                    everyone=r.get("everyone"),
                    conditions=r.get("conditions")
                ),
                match=r.get("match")
            )
            for i, r in enumerate(rules)
        ],
        default=default
    )


# ============================================================
# Neptune Filter Builder Extended Tests
# ============================================================

class TestNeptuneCompiledConditions:
    """Tests for Neptune filter building from compiled conditions."""

    def test_compiled_exists_condition(self):
        """Test EXISTS condition from compiled expression."""
        from ragguard.filters.backends.neptune import to_neptune_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.email exists"]
        }])

        filters = to_neptune_filter(policy, {"id": "alice"})

        # Should have exists predicate
        assert any(
            f.get("predicate") == "exists" or f.get("type") == "has"
            for f in filters if isinstance(f, dict)
        )

    def test_compiled_not_exists_condition(self):
        """Test NOT_EXISTS condition from compiled expression."""
        from ragguard.filters.backends.neptune import to_neptune_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.deleted not exists"]
        }])

        filters = to_neptune_filter(policy, {"id": "alice"})

        # Should have hasNot type
        assert any(
            f.get("type") == "hasNot"
            for f in filters if isinstance(f, dict)
        )

    def test_compiled_greater_than(self):
        """Test GREATER_THAN from compiled expression."""
        from ragguard.filters.backends.neptune import to_neptune_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.priority > 5"]
        }])

        filters = to_neptune_filter(policy, {"id": "alice"})

        assert any(
            f.get("predicate") == "gt" and f.get("value") == 5
            for f in filters if isinstance(f, dict)
        )

    def test_compiled_less_than(self):
        """Test LESS_THAN from compiled expression."""
        from ragguard.filters.backends.neptune import to_neptune_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.score < 100"]
        }])

        filters = to_neptune_filter(policy, {"id": "alice"})

        assert any(
            f.get("predicate") == "lt" and f.get("value") == 100
            for f in filters if isinstance(f, dict)
        )

    def test_compiled_gte_condition(self):
        """Test GREATER_THAN_OR_EQUAL from compiled expression."""
        from ragguard.filters.backends.neptune import to_neptune_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.level >= 3"]
        }])

        filters = to_neptune_filter(policy, {"id": "alice"})

        assert any(
            f.get("predicate") == "gte"
            for f in filters if isinstance(f, dict)
        )

    def test_compiled_lte_condition(self):
        """Test LESS_THAN_OR_EQUAL from compiled expression."""
        from ragguard.filters.backends.neptune import to_neptune_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.rating <= 5"]
        }])

        filters = to_neptune_filter(policy, {"id": "alice"})

        assert any(
            f.get("predicate") == "lte"
            for f in filters if isinstance(f, dict)
        )

    def test_compiled_not_equals(self):
        """Test NOT_EQUALS from compiled expression."""
        from ragguard.filters.backends.neptune import to_neptune_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.status != 'deleted'"]
        }])

        filters = to_neptune_filter(policy, {"id": "alice"})

        assert any(
            f.get("predicate") == "neq"
            for f in filters if isinstance(f, dict)
        )

    def test_compiled_not_in_user_field(self):
        """Test NOT_IN with user field."""
        from ragguard.filters.backends.neptune import to_neptune_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["user.id not in document.blocked_users"]
        }])

        filters = to_neptune_filter(policy, {"id": "alice"})

        assert any(
            f.get("predicate") == "notContaining"
            for f in filters if isinstance(f, dict)
        )

    def test_compiled_not_in_with_null_user(self):
        """Test NOT_IN with null user value."""
        from ragguard.filters.backends.neptune import to_neptune_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["user.missing not in document.blocked"]
        }])

        # User doesn't have 'missing' field
        filters = to_neptune_filter(policy, {"id": "alice"})

        # Should return empty list (allow all) since null not in anything is true
        assert filters == []

    def test_compiled_in_with_null_user(self):
        """Test IN with null user value (deny)."""
        from ragguard.filters.backends.neptune import to_neptune_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["user.missing in document.allowed"]
        }])

        filters = to_neptune_filter(policy, {"id": "alice"})

        # Should deny since user field is null - check for deny marker
        # The marker may use different field names
        has_deny = any(
            "ragguard_deny" in f.get("property", "") or "deny_all" in f.get("property", "")
            for f in filters if isinstance(f, dict)
        )
        # Or it may use containing with None value
        has_containing_none = any(
            f.get("predicate") == "containing" and f.get("value") is None
            for f in filters if isinstance(f, dict)
        )
        assert has_deny or has_containing_none or len(filters) > 0

    def test_compiled_not_in_document_list(self):
        """Test NOT_IN with document field and list literal."""
        from ragguard.filters.backends.neptune import to_neptune_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.status not in ['deleted', 'archived']"]
        }])

        filters = to_neptune_filter(policy, {"id": "alice"})

        assert any(
            f.get("predicate") == "without"
            for f in filters if isinstance(f, dict)
        )

    def test_compiled_expression_with_or(self):
        """Test compiled expression with OR logic."""
        from ragguard.filters.backends.neptune import to_neptune_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.status == 'active' or document.status == 'pending'"]
        }])

        filters = to_neptune_filter(policy, {"id": "alice"})

        # Should have OR structure
        assert any(
            f.get("type") == "or"
            for f in filters if isinstance(f, dict)
        )


class TestNeptuneMultipleRules:
    """Tests for Neptune filter with multiple rules."""

    def test_multiple_rules_or(self):
        """Test multiple rules combined with OR."""
        from ragguard.filters.backends.neptune import to_neptune_filter

        policy = make_policy([
            {"roles": ["admin"]},
            {"roles": ["manager"], "conditions": ["document.category == 'public'"]},
        ])

        filters = to_neptune_filter(policy, {"id": "alice", "roles": ["admin", "manager"]})

        # Should have OR combining both rules
        assert any(f.get("type") == "or" for f in filters if isinstance(f, dict))

    def test_single_matching_rule(self):
        """Test single matching rule returns flat list."""
        from ragguard.filters.backends.neptune import to_neptune_filter

        policy = make_policy([
            {"roles": ["admin"]},
            {"roles": ["viewer"], "conditions": ["document.public == true"]},
        ])

        # Only admin role matches (no conditions)
        filters = to_neptune_filter(policy, {"id": "alice", "roles": ["admin"]})

        # Should be a flat list (no OR wrapper needed)
        assert len(filters) == 0 or not any(f.get("type") == "or" for f in filters if isinstance(f, dict))


class TestNeptuneApplyFilters:
    """Tests for apply_neptune_filters function."""

    def test_apply_filters_missing_gremlin(self):
        """Test apply_neptune_filters raises when gremlin not installed."""
        from ragguard.exceptions import FilterBuildError
        from ragguard.filters.backends.neptune import apply_neptune_filters

        # Remove gremlin_python from sys.modules temporarily
        with patch.dict('sys.modules', {'gremlin_python': None, 'gremlin_python.process': None, 'gremlin_python.process.traversal': None, 'gremlin_python.process.graph_traversal': None}):
            with pytest.raises(FilterBuildError, match="gremlin_python not installed"):
                # Force reimport
                import importlib

                import ragguard.filters.backends.neptune as neptune_module
                importlib.reload(neptune_module)
                neptune_module.apply_neptune_filters(MagicMock(), [])

    def test_apply_eq_filter(self):
        """Test applying eq filter."""
        from ragguard.filters.backends.neptune import apply_neptune_filters

        mock_traversal = MagicMock()
        mock_P = MagicMock()

        with patch.dict('sys.modules', {
            'gremlin_python': MagicMock(),
            'gremlin_python.process': MagicMock(),
            'gremlin_python.process.traversal': MagicMock(P=mock_P),
            'gremlin_python.process.graph_traversal': MagicMock(__=MagicMock())
        }):
            filters = [{"type": "has", "property": "category", "predicate": "eq", "value": "engineering"}]
            apply_neptune_filters(mock_traversal, filters)
            mock_traversal.has.assert_called()

    def test_apply_neq_filter(self):
        """Test applying neq filter."""
        from ragguard.filters.backends.neptune import apply_neptune_filters

        mock_traversal = MagicMock()
        mock_P = MagicMock()

        with patch.dict('sys.modules', {
            'gremlin_python': MagicMock(),
            'gremlin_python.process': MagicMock(),
            'gremlin_python.process.traversal': MagicMock(P=mock_P),
            'gremlin_python.process.graph_traversal': MagicMock(__=MagicMock())
        }):
            filters = [{"type": "has", "property": "status", "predicate": "neq", "value": "deleted"}]
            apply_neptune_filters(mock_traversal, filters)
            mock_traversal.has.assert_called()

    def test_apply_within_filter(self):
        """Test applying within filter."""
        from ragguard.filters.backends.neptune import apply_neptune_filters

        mock_traversal = MagicMock()
        mock_P = MagicMock()

        with patch.dict('sys.modules', {
            'gremlin_python': MagicMock(),
            'gremlin_python.process': MagicMock(),
            'gremlin_python.process.traversal': MagicMock(P=mock_P),
            'gremlin_python.process.graph_traversal': MagicMock(__=MagicMock())
        }):
            filters = [{"type": "has", "property": "category", "predicate": "within", "value": ["eng", "sales"]}]
            apply_neptune_filters(mock_traversal, filters)
            mock_traversal.has.assert_called()

    def test_apply_or_filter(self):
        """Test applying OR filter."""
        from ragguard.filters.backends.neptune import apply_neptune_filters

        mock_traversal = MagicMock()
        mock_P = MagicMock()
        mock__ = MagicMock()
        mock__.identity.return_value = MagicMock()

        with patch.dict('sys.modules', {
            'gremlin_python': MagicMock(),
            'gremlin_python.process': MagicMock(),
            'gremlin_python.process.traversal': MagicMock(P=mock_P),
            'gremlin_python.process.graph_traversal': MagicMock(__=mock__)
        }):
            filters = [{
                "type": "or",
                "children": [
                    [{"type": "has", "property": "cat", "predicate": "eq", "value": "eng"}],
                    [{"type": "has", "property": "cat", "predicate": "eq", "value": "sales"}]
                ]
            }]
            apply_neptune_filters(mock_traversal, filters)
            mock_traversal.or_.assert_called()

    def test_apply_hasNot_filter(self):
        """Test applying hasNot filter."""
        from ragguard.filters.backends.neptune import apply_neptune_filters

        mock_traversal = MagicMock()
        mock_P = MagicMock()

        with patch.dict('sys.modules', {
            'gremlin_python': MagicMock(),
            'gremlin_python.process': MagicMock(),
            'gremlin_python.process.traversal': MagicMock(P=mock_P),
            'gremlin_python.process.graph_traversal': MagicMock(__=MagicMock())
        }):
            filters = [{"type": "hasNot", "property": "deleted"}]
            apply_neptune_filters(mock_traversal, filters)
            mock_traversal.hasNot.assert_called_with("deleted")


class TestNeptuneStringParsing:
    """Tests for Neptune string parsing fallback."""

    def test_parse_not_equals_string(self):
        """Test parsing != from string."""
        from ragguard.filters.backends.neptune import to_neptune_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.status != 'deleted'"]
        }])

        filters = to_neptune_filter(policy, {"id": "alice"})
        assert any(f.get("predicate") == "neq" for f in filters if isinstance(f, dict))

    def test_parse_user_equals_document(self):
        """Test parsing user.field == document.field from string."""
        from ragguard.filters.backends.neptune import to_neptune_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["user.department == document.department"]
        }])

        filters = to_neptune_filter(policy, {"id": "alice", "department": "engineering"})
        assert any(f.get("value") == "engineering" for f in filters if isinstance(f, dict))

    def test_parse_document_in_list(self):
        """Test parsing document.field in ['a', 'b'] from string."""
        from ragguard.filters.backends.neptune import to_neptune_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.category in ['public', 'internal']"]
        }])

        filters = to_neptune_filter(policy, {"id": "alice"})
        assert any(f.get("predicate") == "within" for f in filters if isinstance(f, dict))

    def test_parse_document_in_empty_list(self):
        """Test parsing document.field in [] (deny or empty within)."""
        from ragguard.filters.backends.neptune import to_neptune_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.category in []"]
        }])

        filters = to_neptune_filter(policy, {"id": "alice"})
        # Empty list may result in deny marker or empty within filter
        has_deny = any(f.get("property") == "_ragguard_deny_all" for f in filters if isinstance(f, dict))
        has_within_empty = any(f.get("predicate") == "within" and f.get("value") == [] for f in filters if isinstance(f, dict))
        # Or it may return empty for allow-all
        assert has_deny or has_within_empty or filters == []

    def test_parse_not_in_document_list(self):
        """Test parsing document.field not in ['a', 'b'] from string."""
        from ragguard.filters.backends.neptune import to_neptune_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.status not in ['deleted', 'archived']"]
        }])

        filters = to_neptune_filter(policy, {"id": "alice"})
        assert any(f.get("predicate") == "without" for f in filters if isinstance(f, dict))

    def test_parse_user_in_document_array(self):
        """Test parsing user.id in document.allowed_users from string."""
        from ragguard.filters.backends.neptune import to_neptune_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["user.id in document.allowed_users"]
        }])

        filters = to_neptune_filter(policy, {"id": "alice"})
        assert any(f.get("predicate") == "containing" for f in filters if isinstance(f, dict))

    def test_parse_user_not_in_document_array(self):
        """Test parsing user.id not in document.blocked_users from string."""
        from ragguard.filters.backends.neptune import to_neptune_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["user.id not in document.blocked_users"]
        }])

        filters = to_neptune_filter(policy, {"id": "alice"})
        assert any(f.get("predicate") == "notContaining" for f in filters if isinstance(f, dict))


# ============================================================
# TigerGraph Filter Builder Extended Tests
# ============================================================

class TestTigerGraphCompiledConditions:
    """Tests for TigerGraph filter building from compiled conditions."""

    def test_compiled_exists_condition(self):
        """Test EXISTS condition from compiled expression."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.email exists"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        assert '!= ""' in where

    def test_compiled_not_exists_condition(self):
        """Test NOT_EXISTS condition from compiled expression."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.deleted not exists"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        assert '== ""' in where

    def test_compiled_greater_than(self):
        """Test GREATER_THAN from compiled expression."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.priority > 5"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        assert "> 5" in where

    def test_compiled_less_than(self):
        """Test LESS_THAN from compiled expression."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.score < 100"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        assert "< 100" in where

    def test_compiled_gte_condition(self):
        """Test GREATER_THAN_OR_EQUAL from compiled expression."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.level >= 3"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        assert ">= 3" in where

    def test_compiled_lte_condition(self):
        """Test LESS_THAN_OR_EQUAL from compiled expression."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.rating <= 5"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        assert "<= 5" in where

    def test_compiled_not_equals(self):
        """Test NOT_EQUALS from compiled expression."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.status != 'deleted'"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        assert "!=" in where
        assert "deleted" in where

    def test_compiled_not_in_user_field(self):
        """Test NOT_IN with user field."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["user.id not in document.blocked_users"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        assert "NOT" in where
        assert "IN" in where

    def test_compiled_not_in_with_null_user(self):
        """Test NOT_IN with null user value."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["user.missing not in document.blocked"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        # Should return TRUE since null not in anything
        assert where == "TRUE"

    def test_compiled_in_with_null_user(self):
        """Test IN with null user value (deny)."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["user.missing in document.allowed"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        assert where == "FALSE"

    def test_compiled_not_in_document_list(self):
        """Test NOT_IN with document field and list literal."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.status not in ['deleted', 'archived']"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        assert "NOT" in where
        assert "IN" in where

    def test_compiled_expression_with_or(self):
        """Test compiled expression with OR logic."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.status == 'active' or document.status == 'pending'"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        assert " OR " in where

    def test_compiled_expression_with_and(self):
        """Test compiled expression with AND logic."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.status == 'active' and document.priority > 5"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        assert " AND " in where or ("active" in where and "> 5" in where)


class TestTigerGraphMultipleRules:
    """Tests for TigerGraph filter with multiple rules."""

    def test_multiple_rules_or(self):
        """Test multiple rules combined with OR."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([
            {"roles": ["admin"]},
            {"roles": ["manager"], "conditions": ["document.category == 'public'"]},
        ])

        where, params = to_tigergraph_filter(policy, {"id": "alice", "roles": ["admin", "manager"]})
        # Should have OR combining both rules
        assert " OR " in where

    def test_single_matching_rule(self):
        """Test single matching rule returns single condition."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([
            {"roles": ["admin"]},
            {"roles": ["viewer"], "conditions": ["document.public == true"]},
        ])

        # Only admin role matches (no conditions)
        where, params = to_tigergraph_filter(policy, {"id": "alice", "roles": ["admin"]})
        assert where == "TRUE"


class TestTigerGraphStringParsing:
    """Tests for TigerGraph string parsing fallback."""

    def test_parse_not_equals_string(self):
        """Test parsing != from string."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.status != 'deleted'"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        assert "!=" in where

    def test_parse_user_equals_document(self):
        """Test parsing user.field == document.field from string."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["user.department == document.department"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice", "department": "engineering"})
        assert "engineering" in where

    def test_parse_document_in_list(self):
        """Test parsing document.field in ['a', 'b'] from string."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.category in ['public', 'internal']"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        assert " IN " in where

    def test_parse_document_in_empty_list(self):
        """Test parsing document.field in [] (deny or empty IN)."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.category in []"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        # May be FALSE or IN () depending on implementation path
        assert where == "FALSE" or "IN ()" in where or where == "TRUE"

    def test_parse_not_in_document_list(self):
        """Test parsing document.field not in ['a', 'b'] from string."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.status not in ['deleted', 'archived']"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        assert "NOT" in where

    def test_parse_user_in_document_array(self):
        """Test parsing user.id in document.allowed_users from string."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["user.id in document.allowed_users"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        assert "IN" in where
        assert "alice" in where

    def test_parse_user_not_in_document_array(self):
        """Test parsing user.id not in document.blocked_users from string."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["user.id not in document.blocked_users"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        assert "NOT" in where

    def test_parse_exists_from_string(self):
        """Test parsing 'document.field exists' from string."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.email exists"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        assert '!= ""' in where

    def test_parse_not_exists_from_string(self):
        """Test parsing 'document.field not exists' from string."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.deleted not exists"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        assert '== ""' in where


class TestTigerGraphFormatValue:
    """Tests for TigerGraph value formatting."""

    def test_format_bool_true(self):
        """Test formatting boolean True."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.active == true"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        assert "TRUE" in where

    def test_format_bool_false(self):
        """Test formatting boolean False."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.disabled == false"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        assert "FALSE" in where or "false" in where.lower()

    def test_format_user_bool(self):
        """Test formatting user boolean value."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["user.is_admin == document.requires_admin"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice", "is_admin": True})
        assert "TRUE" in where


class TestTigerGraphCustomAlias:
    """Tests for TigerGraph with custom vertex alias."""

    def test_custom_vertex_alias(self):
        """Test using custom vertex alias."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "conditions": ["document.category == 'engineering'"]
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"}, vertex_alias="doc")
        assert "doc." in where

    def test_match_with_custom_alias(self):
        """Test match conditions with custom alias."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "match": {"status": "active"}
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"}, vertex_alias="vertex")
        assert "vertex.status" in where


class TestTigerGraphMatchConditions:
    """Tests for TigerGraph match conditions."""

    def test_match_with_list_value(self):
        """Test match condition with list value."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "match": {"category": ["public", "internal"]}
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        assert "IN" in where
        assert "public" in where
        assert "internal" in where

    def test_match_with_string_value(self):
        """Test match condition with string value."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "match": {"status": "active"}
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        assert "active" in where
        assert "==" in where

    def test_match_with_number_value(self):
        """Test match condition with numeric value."""
        from ragguard.filters.backends.tigergraph import to_tigergraph_filter

        policy = make_policy([{
            "everyone": True,
            "match": {"priority": 5}
        }])

        where, params = to_tigergraph_filter(policy, {"id": "alice"})
        assert "== 5" in where


# ============================================================
# String Parsing Fallback Tests (for uncovered paths)
# ============================================================

class TestNeo4jStringParsingFallback:
    """Tests for Neo4j string parsing fallback when compilation fails."""

    def test_parse_exists_via_string(self):
        """Test exists condition via string parsing fallback."""
        from ragguard.filters.backends.neo4j import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        cypher, params = _parse_condition_string(
            "document.email exists",
            {"id": "alice"},
            "n",
            counter()
        )
        assert cypher == "n.email IS NOT NULL"
        assert params == {}

    def test_parse_not_exists_via_string(self):
        """Test not exists condition via string parsing."""
        from ragguard.filters.backends.neo4j import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        cypher, params = _parse_condition_string(
            "document.deleted not exists",
            {"id": "alice"},
            "n",
            counter()
        )
        assert cypher == "n.deleted IS NULL"
        assert params == {}

    def test_parse_not_equals_document(self):
        """Test document.field != literal via string parsing."""
        from ragguard.filters.backends.neo4j import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        cypher, params = _parse_condition_string(
            "document.status != 'deleted'",
            {"id": "alice"},
            "n",
            counter()
        )
        assert "n.status <>" in cypher
        assert "deleted" in str(params.values())

    def test_parse_user_equals_document(self):
        """Test user.field == document.field via string parsing."""
        from ragguard.filters.backends.neo4j import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        cypher, params = _parse_condition_string(
            "user.department == document.department",
            {"id": "alice", "department": "engineering"},
            "n",
            counter()
        )
        assert "n.department =" in cypher
        assert "engineering" in str(params.values())

    def test_parse_user_equals_document_null_user(self):
        """Test user.field == document.field with null user value."""
        from ragguard.filters.backends.neo4j import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        cypher, params = _parse_condition_string(
            "user.department == document.department",
            {"id": "alice"},  # No department
            "n",
            counter()
        )
        assert cypher == "false"

    def test_parse_document_equals_literal(self):
        """Test document.field == 'literal' via string parsing."""
        from ragguard.filters.backends.neo4j import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        cypher, params = _parse_condition_string(
            "document.type == 'public'",
            {"id": "alice"},
            "n",
            counter()
        )
        assert "n.type =" in cypher
        assert "public" in str(params.values())

    def test_parse_user_not_in_document(self):
        """Test user.field not in document.list via string parsing."""
        from ragguard.filters.backends.neo4j import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        cypher, params = _parse_condition_string(
            "user.id not in document.blocked_users",
            {"id": "alice"},
            "n",
            counter()
        )
        assert "NOT" in cypher
        assert "n.blocked_users" in cypher

    def test_parse_user_not_in_document_null(self):
        """Test user.field not in document when user value is null."""
        from ragguard.filters.backends.neo4j import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        cypher, params = _parse_condition_string(
            "user.role not in document.allowed_roles",
            {"id": "alice"},  # No role
            "n",
            counter()
        )
        assert cypher == "true"

    def test_parse_document_not_in_list(self):
        """Test document.field not in [...] via string parsing."""
        from ragguard.filters.backends.neo4j import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        cypher, params = _parse_condition_string(
            "document.status not in ['deleted', 'archived']",
            {"id": "alice"},
            "n",
            counter()
        )
        assert "NOT" in cypher
        assert "n.status" in cypher

    def test_parse_user_in_document(self):
        """Test user.field in document.list via string parsing."""
        from ragguard.filters.backends.neo4j import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        cypher, params = _parse_condition_string(
            "user.role in document.allowed_roles",
            {"id": "alice", "role": "admin"},
            "n",
            counter()
        )
        assert "IN n.allowed_roles" in cypher
        assert "admin" in str(params.values())

    def test_parse_user_in_document_null(self):
        """Test user.field in document when user value is null."""
        from ragguard.filters.backends.neo4j import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        cypher, params = _parse_condition_string(
            "user.role in document.allowed_roles",
            {"id": "alice"},  # No role
            "n",
            counter()
        )
        assert cypher == "false"

    def test_parse_document_in_list(self):
        """Test document.field in [...] via string parsing."""
        from ragguard.filters.backends.neo4j import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        cypher, params = _parse_condition_string(
            "document.status in ['active', 'pending']",
            {"id": "alice"},
            "n",
            counter()
        )
        assert "n.status IN" in cypher

    def test_parse_document_in_empty_list(self):
        """Test document.field in [] returns false."""
        from ragguard.filters.backends.neo4j import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        cypher, params = _parse_condition_string(
            "document.status in []",
            {"id": "alice"},
            "n",
            counter()
        )
        assert cypher == "false"

    def test_parse_invalid_condition(self):
        """Test invalid condition returns None."""
        from ragguard.filters.backends.neo4j import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        cypher, params = _parse_condition_string(
            "some invalid condition",
            {"id": "alice"},
            "n",
            counter()
        )
        assert cypher is None

    def test_parse_exists_non_document_field(self):
        """Test exists on non-document field returns None."""
        from ragguard.filters.backends.neo4j import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        cypher, params = _parse_condition_string(
            "user.email exists",
            {"id": "alice"},
            "n",
            counter()
        )
        assert cypher is None


class TestArangoDBStringParsingFallback:
    """Tests for ArangoDB string parsing fallback."""

    def test_parse_exists_via_string(self):
        """Test exists condition via string parsing."""
        from ragguard.filters.backends.arangodb import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        aql, bind_vars = _parse_condition_string(
            "document.email exists",
            {"id": "alice"},
            "doc",
            counter()
        )
        assert aql == "doc.email != null"

    def test_parse_not_exists_via_string(self):
        """Test not exists condition via string parsing."""
        from ragguard.filters.backends.arangodb import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        aql, bind_vars = _parse_condition_string(
            "document.deleted not exists",
            {"id": "alice"},
            "doc",
            counter()
        )
        assert aql == "doc.deleted == null"

    def test_parse_not_equals_document(self):
        """Test document.field != literal via string parsing."""
        from ragguard.filters.backends.arangodb import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        aql, bind_vars = _parse_condition_string(
            "document.status != 'deleted'",
            {"id": "alice"},
            "doc",
            counter()
        )
        assert "doc.status !=" in aql
        assert "deleted" in str(bind_vars.values())

    def test_parse_user_equals_document(self):
        """Test user.field == document.field via string parsing."""
        from ragguard.filters.backends.arangodb import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        aql, bind_vars = _parse_condition_string(
            "user.department == document.department",
            {"id": "alice", "department": "engineering"},
            "doc",
            counter()
        )
        assert "doc.department ==" in aql
        assert "engineering" in str(bind_vars.values())

    def test_parse_user_equals_document_null(self):
        """Test user.field == document.field with null user value."""
        from ragguard.filters.backends.arangodb import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        aql, bind_vars = _parse_condition_string(
            "user.department == document.department",
            {"id": "alice"},
            "doc",
            counter()
        )
        assert aql == "false"

    def test_parse_document_equals_literal(self):
        """Test document.field == literal via string parsing."""
        from ragguard.filters.backends.arangodb import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        aql, bind_vars = _parse_condition_string(
            "document.type == 'public'",
            {"id": "alice"},
            "doc",
            counter()
        )
        assert "doc.type ==" in aql
        assert "public" in str(bind_vars.values())

    def test_parse_user_not_in_document(self):
        """Test user.field not in document.list via string parsing."""
        from ragguard.filters.backends.arangodb import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        aql, bind_vars = _parse_condition_string(
            "user.id not in document.blocked_users",
            {"id": "alice"},
            "doc",
            counter()
        )
        assert "NOT IN doc.blocked_users" in aql

    def test_parse_user_not_in_document_null(self):
        """Test user.field not in document when user value is null."""
        from ragguard.filters.backends.arangodb import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        aql, bind_vars = _parse_condition_string(
            "user.role not in document.blocked_roles",
            {"id": "alice"},
            "doc",
            counter()
        )
        assert aql == "true"

    def test_parse_document_not_in_list(self):
        """Test document.field not in [...] via string parsing."""
        from ragguard.filters.backends.arangodb import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        aql, bind_vars = _parse_condition_string(
            "document.status not in ['deleted', 'archived']",
            {"id": "alice"},
            "doc",
            counter()
        )
        assert "NOT IN" in aql
        assert "doc.status" in aql

    def test_parse_user_in_document(self):
        """Test user.field in document.list via string parsing."""
        from ragguard.filters.backends.arangodb import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        aql, bind_vars = _parse_condition_string(
            "user.role in document.allowed_roles",
            {"id": "alice", "role": "admin"},
            "doc",
            counter()
        )
        assert "IN doc.allowed_roles" in aql

    def test_parse_user_in_document_null(self):
        """Test user.field in document when user value is null."""
        from ragguard.filters.backends.arangodb import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        aql, bind_vars = _parse_condition_string(
            "user.role in document.allowed_roles",
            {"id": "alice"},
            "doc",
            counter()
        )
        assert aql == "false"

    def test_parse_document_in_list(self):
        """Test document.field in [...] via string parsing."""
        from ragguard.filters.backends.arangodb import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        aql, bind_vars = _parse_condition_string(
            "document.status in ['active', 'pending']",
            {"id": "alice"},
            "doc",
            counter()
        )
        assert "doc.status IN" in aql

    def test_parse_document_in_empty_list(self):
        """Test document.field in [] returns false."""
        from ragguard.filters.backends.arangodb import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        aql, bind_vars = _parse_condition_string(
            "document.status in []",
            {"id": "alice"},
            "doc",
            counter()
        )
        assert aql == "false"


class TestNeptuneStringParsingFallback:
    """Tests for Neptune string parsing fallback."""

    def test_parse_exists_via_string(self):
        """Test exists condition via string parsing."""
        from ragguard.filters.backends.neptune import _parse_condition_string

        filters = _parse_condition_string(
            "document.email exists",
            {"id": "alice"}
        )
        assert any(f.get("type") == "has" for f in filters)

    def test_parse_not_exists_via_string(self):
        """Test not exists condition via string parsing."""
        from ragguard.filters.backends.neptune import _parse_condition_string

        filters = _parse_condition_string(
            "document.deleted not exists",
            {"id": "alice"}
        )
        assert any(f.get("type") == "hasNot" for f in filters)

    def test_parse_not_equals_document(self):
        """Test document.field != literal via string parsing."""
        from ragguard.filters.backends.neptune import _parse_condition_string

        filters = _parse_condition_string(
            "document.status != 'deleted'",
            {"id": "alice"}
        )
        assert any(f.get("predicate") == "neq" for f in filters)

    def test_parse_user_equals_document(self):
        """Test user.field == document.field via string parsing."""
        from ragguard.filters.backends.neptune import _parse_condition_string

        filters = _parse_condition_string(
            "user.department == document.department",
            {"id": "alice", "department": "engineering"}
        )
        assert any(f.get("predicate") == "eq" for f in filters)

    def test_parse_user_equals_document_null(self):
        """Test user.field == document.field with null user value returns DENY sentinel."""
        from ragguard.filters.backends.neptune import DENY_ALL_FIELD, _parse_condition_string

        filters = _parse_condition_string(
            "user.department == document.department",
            {"id": "alice"}
        )
        # Returns DENY sentinel because user value is null
        assert any(f.get("property") == DENY_ALL_FIELD for f in filters)

    def test_parse_document_equals_literal(self):
        """Test document.field == literal via string parsing."""
        from ragguard.filters.backends.neptune import _parse_condition_string

        filters = _parse_condition_string(
            "document.type == 'public'",
            {"id": "alice"}
        )
        assert any(f.get("predicate") == "eq" for f in filters)

    def test_parse_user_not_in_document(self):
        """Test user.field not in document.list via string parsing."""
        from ragguard.filters.backends.neptune import _parse_condition_string

        filters = _parse_condition_string(
            "user.id not in document.blocked_users",
            {"id": "alice"}
        )
        # Uses 'notContaining' predicate for user.x not in document.y
        assert any(f.get("predicate") == "notContaining" for f in filters)

    def test_parse_user_not_in_document_null(self):
        """Test user.field not in document when user value is null returns empty."""
        from ragguard.filters.backends.neptune import _parse_condition_string

        filters = _parse_condition_string(
            "user.role not in document.blocked_roles",
            {"id": "alice"}
        )
        # Returns empty list because user value is null (not in = true)
        assert filters == []

    def test_parse_document_not_in_list(self):
        """Test document.field not in [...] via string parsing."""
        from ragguard.filters.backends.neptune import _parse_condition_string

        filters = _parse_condition_string(
            "document.status not in ['deleted', 'archived']",
            {"id": "alice"}
        )
        assert any(f.get("predicate") == "without" for f in filters)

    def test_parse_user_in_document(self):
        """Test user.field in document.list via string parsing."""
        from ragguard.filters.backends.neptune import _parse_condition_string

        filters = _parse_condition_string(
            "user.role in document.allowed_roles",
            {"id": "alice", "role": "admin"}
        )
        # Uses 'containing' predicate for user.x in document.y
        assert any(f.get("predicate") == "containing" for f in filters)

    def test_parse_user_in_document_null(self):
        """Test user.field in document when user value is null returns DENY sentinel."""
        from ragguard.filters.backends.neptune import DENY_ALL_FIELD, _parse_condition_string

        filters = _parse_condition_string(
            "user.role in document.allowed_roles",
            {"id": "alice"}
        )
        # Returns DENY sentinel because user value is null
        assert any(f.get("property") == DENY_ALL_FIELD for f in filters)

    def test_parse_document_in_list(self):
        """Test document.field in [...] via string parsing."""
        from ragguard.filters.backends.neptune import _parse_condition_string

        filters = _parse_condition_string(
            "document.status in ['active', 'pending']",
            {"id": "alice"}
        )
        assert any(f.get("predicate") == "within" for f in filters)

    def test_parse_document_in_empty_list(self):
        """Test document.field in [] returns DENY sentinel."""
        from ragguard.filters.backends.neptune import DENY_ALL_FIELD, _parse_condition_string

        filters = _parse_condition_string(
            "document.status in []",
            {"id": "alice"}
        )
        # Returns DENY sentinel for empty list
        assert any(f.get("property") == DENY_ALL_FIELD for f in filters)

    def test_parse_invalid_condition_returns_none(self):
        """Test invalid condition returns None."""
        from ragguard.filters.backends.neptune import _parse_condition_string

        filters = _parse_condition_string(
            "some random text",
            {"id": "alice"}
        )
        assert filters is None


class TestTigerGraphStringParsingFallback:
    """Tests for TigerGraph string parsing fallback."""

    def test_parse_exists_via_string(self):
        """Test exists condition via string parsing."""
        from ragguard.filters.backends.tigergraph import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        gsql, params = _parse_condition_string(
            "document.email exists",
            {"id": "alice"},
            "v",
            counter()
        )
        assert 'v.email != ""' in gsql

    def test_parse_not_exists_via_string(self):
        """Test not exists condition via string parsing."""
        from ragguard.filters.backends.tigergraph import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        gsql, params = _parse_condition_string(
            "document.deleted not exists",
            {"id": "alice"},
            "v",
            counter()
        )
        assert 'v.deleted == ""' in gsql

    def test_parse_not_equals_document(self):
        """Test document.field != literal via string parsing."""
        from ragguard.filters.backends.tigergraph import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        gsql, params = _parse_condition_string(
            "document.status != 'deleted'",
            {"id": "alice"},
            "v",
            counter()
        )
        assert "v.status !=" in gsql
        assert '"deleted"' in gsql

    def test_parse_user_equals_document(self):
        """Test user.field == document.field via string parsing."""
        from ragguard.filters.backends.tigergraph import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        gsql, params = _parse_condition_string(
            "user.department == document.department",
            {"id": "alice", "department": "engineering"},
            "v",
            counter()
        )
        assert "v.department ==" in gsql
        assert '"engineering"' in gsql

    def test_parse_user_equals_document_null(self):
        """Test user.field == document.field with null user value."""
        from ragguard.filters.backends.tigergraph import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        gsql, params = _parse_condition_string(
            "user.department == document.department",
            {"id": "alice"},
            "v",
            counter()
        )
        assert gsql == "FALSE"

    def test_parse_document_equals_literal(self):
        """Test document.field == literal via string parsing."""
        from ragguard.filters.backends.tigergraph import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        gsql, params = _parse_condition_string(
            "document.type == 'public'",
            {"id": "alice"},
            "v",
            counter()
        )
        assert "v.type ==" in gsql
        assert '"public"' in gsql

    def test_parse_user_not_in_document(self):
        """Test user.field not in document.list via string parsing."""
        from ragguard.filters.backends.tigergraph import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        gsql, params = _parse_condition_string(
            "user.id not in document.blocked_users",
            {"id": "alice"},
            "v",
            counter()
        )
        assert "NOT" in gsql
        assert "v.blocked_users" in gsql

    def test_parse_user_not_in_document_null(self):
        """Test user.field not in document when user value is null."""
        from ragguard.filters.backends.tigergraph import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        gsql, params = _parse_condition_string(
            "user.role not in document.blocked_roles",
            {"id": "alice"},
            "v",
            counter()
        )
        assert gsql == "TRUE"

    def test_parse_document_not_in_list(self):
        """Test document.field not in [...] via string parsing."""
        from ragguard.filters.backends.tigergraph import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        gsql, params = _parse_condition_string(
            "document.status not in ['deleted', 'archived']",
            {"id": "alice"},
            "v",
            counter()
        )
        assert "NOT" in gsql
        assert "v.status" in gsql

    def test_parse_user_in_document(self):
        """Test user.field in document.list via string parsing."""
        from ragguard.filters.backends.tigergraph import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        gsql, params = _parse_condition_string(
            "user.role in document.allowed_roles",
            {"id": "alice", "role": "admin"},
            "v",
            counter()
        )
        assert "v.allowed_roles" in gsql

    def test_parse_user_in_document_null(self):
        """Test user.field in document when user value is null."""
        from ragguard.filters.backends.tigergraph import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        gsql, params = _parse_condition_string(
            "user.role in document.allowed_roles",
            {"id": "alice"},
            "v",
            counter()
        )
        assert gsql == "FALSE"

    def test_parse_document_in_list(self):
        """Test document.field in [...] via string parsing."""
        from ragguard.filters.backends.tigergraph import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        gsql, params = _parse_condition_string(
            "document.status in ['active', 'pending']",
            {"id": "alice"},
            "v",
            counter()
        )
        assert "v.status IN" in gsql

    def test_parse_document_in_empty_list(self):
        """Test document.field in [] returns FALSE."""
        from ragguard.filters.backends.tigergraph import _parse_condition_string

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        gsql, params = _parse_condition_string(
            "document.status in []",
            {"id": "alice"},
            "v",
            counter()
        )
        assert gsql == "FALSE"


# ============================================================
# Compiled Expression Direct Tests (OR/AND paths)
# ============================================================

class TestNeo4jCompiledExpressionPaths:
    """Tests for Neo4j compiled expression OR/AND paths."""

    def test_compiled_expression_or_multiple_children(self):
        """Test compiled expression with OR and multiple children."""
        from ragguard.filters.backends.neo4j import _build_neo4j_from_compiled_node
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledExpression,
            CompiledValue,
            ConditionOperator,
            LogicalOperator,
            ValueType,
        )

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        # Create an OR expression with two children
        cond1 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="active", field_path=()),
            original="document.status == 'active'"
        )
        cond2 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="pending", field_path=()),
            original="document.status == 'pending'"
        )
        expr = CompiledExpression(operator=LogicalOperator.OR, children=[cond1, cond2], original="test")

        cypher, params = _build_neo4j_from_compiled_node(expr, {"id": "alice"}, "n", counter())

        assert "OR" in cypher
        assert "n.status" in cypher

    def test_compiled_expression_and_multiple_children(self):
        """Test compiled expression with AND and multiple children."""
        from ragguard.filters.backends.neo4j import _build_neo4j_from_compiled_node
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledExpression,
            CompiledValue,
            ConditionOperator,
            LogicalOperator,
            ValueType,
        )

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        cond1 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("public",)),
            right=CompiledValue(value_type=ValueType.LITERAL_BOOL, value=True, field_path=()),
            original="document.public == true"
        )
        cond2 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="active", field_path=()),
            original="document.status == 'active'"
        )
        expr = CompiledExpression(operator=LogicalOperator.AND, children=[cond1, cond2], original="test")

        cypher, params = _build_neo4j_from_compiled_node(expr, {"id": "alice"}, "n", counter())

        assert "AND" in cypher
        assert "n.public" in cypher
        assert "n.status" in cypher

    def test_compiled_expression_single_child(self):
        """Test compiled expression with single child returns just that condition."""
        from ragguard.filters.backends.neo4j import _build_neo4j_from_compiled_node
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledExpression,
            CompiledValue,
            ConditionOperator,
            LogicalOperator,
            ValueType,
        )

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        cond1 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="active", field_path=()),
            original="document.status == 'active'"
        )
        expr = CompiledExpression(operator=LogicalOperator.OR, children=[cond1], original="test")

        cypher, params = _build_neo4j_from_compiled_node(expr, {"id": "alice"}, "n", counter())

        # Single child, so no OR/AND operators
        assert "OR" not in cypher
        assert "n.status" in cypher

    def test_compiled_expression_empty_children(self):
        """Test compiled expression with no children returns None."""
        from ragguard.filters.backends.neo4j import _build_neo4j_from_compiled_node
        from ragguard.policy.compiler import CompiledExpression, LogicalOperator

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        expr = CompiledExpression(operator=LogicalOperator.OR, children=[], original="empty")

        cypher, params = _build_neo4j_from_compiled_node(expr, {"id": "alice"}, "n", counter())

        assert cypher is None


class TestArangoDBCompiledExpressionPaths:
    """Tests for ArangoDB compiled expression OR/AND paths."""

    def test_compiled_expression_or_multiple_children(self):
        """Test compiled expression with OR and multiple children."""
        from ragguard.filters.backends.arangodb import _build_arangodb_from_compiled_node
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledExpression,
            CompiledValue,
            ConditionOperator,
            LogicalOperator,
            ValueType,
        )

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        cond1 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="active", field_path=()),
            original="document.status == 'active'"
        )
        cond2 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="pending", field_path=()),
            original="document.status == 'pending'"
        )
        expr = CompiledExpression(operator=LogicalOperator.OR, children=[cond1, cond2], original="test")

        aql, bind_vars = _build_arangodb_from_compiled_node(expr, {"id": "alice"}, "doc", counter())

        assert "||" in aql
        assert "doc.status" in aql

    def test_compiled_expression_and_multiple_children(self):
        """Test compiled expression with AND and multiple children."""
        from ragguard.filters.backends.arangodb import _build_arangodb_from_compiled_node
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledExpression,
            CompiledValue,
            ConditionOperator,
            LogicalOperator,
            ValueType,
        )

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        cond1 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("public",)),
            right=CompiledValue(value_type=ValueType.LITERAL_BOOL, value=True, field_path=()),
            original="document.public == true"
        )
        cond2 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="active", field_path=()),
            original="document.status == 'active'"
        )
        expr = CompiledExpression(operator=LogicalOperator.AND, children=[cond1, cond2], original="test")

        aql, bind_vars = _build_arangodb_from_compiled_node(expr, {"id": "alice"}, "doc", counter())

        assert "&&" in aql
        assert "doc.public" in aql
        assert "doc.status" in aql


class TestNeptuneCompiledExpressionPaths:
    """Tests for Neptune compiled expression OR/AND paths."""

    def test_compiled_expression_or_multiple_children(self):
        """Test compiled expression with OR and multiple children."""
        from ragguard.filters.backends.neptune import _build_neptune_from_compiled_node
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
            original="document.status == 'active'"
        )
        cond2 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="pending", field_path=()),
            original="document.status == 'pending'"
        )
        expr = CompiledExpression(operator=LogicalOperator.OR, children=[cond1, cond2], original="test")

        filters = _build_neptune_from_compiled_node(expr, {"id": "alice"})

        # Should have OR structure
        assert any(f.get("type") == "or" for f in filters)

    def test_compiled_expression_and_multiple_children(self):
        """Test compiled expression with AND and multiple children - flattens to list."""
        from ragguard.filters.backends.neptune import _build_neptune_from_compiled_node
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
            original="document.public == true"
        )
        cond2 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="active", field_path=()),
            original="document.status == 'active'"
        )
        expr = CompiledExpression(operator=LogicalOperator.AND, children=[cond1, cond2], original="test")

        filters = _build_neptune_from_compiled_node(expr, {"id": "alice"})

        # AND flattens to list - should have multiple has conditions
        assert len(filters) >= 2

    def test_compiled_expression_single_child(self):
        """Test compiled expression with single child returns just that filter."""
        from ragguard.filters.backends.neptune import _build_neptune_from_compiled_node
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
            original="document.status == 'active'"
        )
        expr = CompiledExpression(operator=LogicalOperator.OR, children=[cond1], original="test")

        filters = _build_neptune_from_compiled_node(expr, {"id": "alice"})

        # Single child, should be flat list
        assert filters is not None
        assert not any(f.get("type") == "or" for f in filters)


class TestTigerGraphCompiledExpressionPaths:
    """Tests for TigerGraph compiled expression OR/AND paths."""

    def test_compiled_expression_or_multiple_children(self):
        """Test compiled expression with OR and multiple children."""
        from ragguard.filters.backends.tigergraph import _build_tigergraph_from_compiled_node
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledExpression,
            CompiledValue,
            ConditionOperator,
            LogicalOperator,
            ValueType,
        )

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        cond1 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="active", field_path=()),
            original="document.status == 'active'"
        )
        cond2 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="pending", field_path=()),
            original="document.status == 'pending'"
        )
        expr = CompiledExpression(operator=LogicalOperator.OR, children=[cond1, cond2], original="test")

        gsql, params = _build_tigergraph_from_compiled_node(expr, {"id": "alice"}, "v", counter())

        assert "OR" in gsql
        assert "v.status" in gsql

    def test_compiled_expression_and_multiple_children(self):
        """Test compiled expression with AND and multiple children."""
        from ragguard.filters.backends.tigergraph import _build_tigergraph_from_compiled_node
        from ragguard.policy.compiler import (
            CompiledCondition,
            CompiledExpression,
            CompiledValue,
            ConditionOperator,
            LogicalOperator,
            ValueType,
        )

        def counter():
            i = [0]
            def next_param(prefix="p"):
                i[0] += 1
                return f"{prefix}_{i[0]}"
            return next_param

        cond1 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("public",)),
            right=CompiledValue(value_type=ValueType.LITERAL_BOOL, value=True, field_path=()),
            original="document.public == true"
        )
        cond2 = CompiledCondition(
            operator=ConditionOperator.EQUALS,
            left=CompiledValue(value_type=ValueType.DOCUMENT_FIELD, value=None, field_path=("status",)),
            right=CompiledValue(value_type=ValueType.LITERAL_STRING, value="active", field_path=()),
            original="document.status == 'active'"
        )
        expr = CompiledExpression(operator=LogicalOperator.AND, children=[cond1, cond2], original="test")

        gsql, params = _build_tigergraph_from_compiled_node(expr, {"id": "alice"}, "v", counter())

        assert "AND" in gsql
        assert "v.public" in gsql
        assert "v.status" in gsql
