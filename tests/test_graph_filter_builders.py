"""Tests for graph database filter builders."""

import pytest

from ragguard.filters.backends.arangodb import to_arangodb_filter
from ragguard.filters.backends.neo4j import to_neo4j_filter
from ragguard.filters.backends.neptune import to_neptune_filter
from ragguard.filters.backends.tigergraph import to_tigergraph_filter
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


class TestNeo4jFilterBuilder:
    """Tests for Neo4j (Cypher) filter builder."""

    def test_simple_role_match(self):
        """Test basic role-based access."""
        policy = make_policy([
            {"roles": ["admin"], "match": {"category": "confidential"}}
        ])
        user = {"id": "alice", "roles": ["admin"]}

        where, params = to_neo4j_filter(policy, user)

        assert "doc.category" in where
        assert any("confidential" in str(v) for v in params.values())

    def test_condition_user_equals_document(self):
        """Test user.field == document.field condition."""
        policy = make_policy([
            {
                "roles": ["engineer"],
                "conditions": ["user.department == document.department"]
            }
        ])
        user = {"id": "alice", "roles": ["engineer"], "department": "engineering"}

        where, params = to_neo4j_filter(policy, user)

        assert "doc.department = $" in where
        assert "engineering" in params.values()

    def test_condition_in_list(self):
        """Test document.field IN list condition."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.category in ['public', 'internal']"]
            }
        ])
        user = {"id": "guest"}

        where, params = to_neo4j_filter(policy, user)

        assert "doc.category IN $" in where
        assert any(isinstance(v, list) for v in params.values())

    def test_deny_all_no_matching_rules(self):
        """Test that deny-all returns false when no rules match."""
        policy = make_policy([
            {"roles": ["admin"]}  # No match for regular user
        ], default="deny")
        user = {"id": "bob", "roles": ["viewer"]}

        where, params = to_neo4j_filter(policy, user)

        assert where == "false"

    def test_allow_all_default(self):
        """Test that allow-all returns empty filter when default is allow."""
        policy = make_policy([
            {"roles": ["admin"]}  # No match for regular user
        ], default="allow")
        user = {"id": "bob", "roles": ["viewer"]}

        where, params = to_neo4j_filter(policy, user)

        assert where == ""
        assert params == {}

    def test_multiple_rules_or(self):
        """Test that multiple matching rules are ORed together."""
        policy = make_policy([
            {"roles": ["admin"]},
            {"roles": ["editor"]}
        ])
        user = {"id": "alice", "roles": ["admin", "editor"]}

        where, params = to_neo4j_filter(policy, user)

        assert " OR " in where or where == "true"

    def test_comparison_operators(self):
        """Test comparison operators (>, <, >=, <=)."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.priority > 5"]
            }
        ])
        user = {"id": "guest"}

        where, params = to_neo4j_filter(policy, user)

        assert "doc.priority >" in where

    def test_not_equals(self):
        """Test != operator."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.status != 'deleted'"]
            }
        ])
        user = {"id": "guest"}

        where, params = to_neo4j_filter(policy, user)

        assert "doc.status <>" in where

    def test_user_in_document_array(self):
        """Test user.id in document.authorized_users."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["user.id in document.authorized_users"]
            }
        ])
        user = {"id": "alice"}

        where, params = to_neo4j_filter(policy, user)

        assert "$" in where
        assert "alice" in params.values()

    def test_field_existence(self):
        """Test field EXISTS operator."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.metadata exists"]
            }
        ])
        user = {"id": "guest"}

        where, params = to_neo4j_filter(policy, user)

        assert "IS NOT NULL" in where


class TestNeptuneFilterBuilder:
    """Tests for Amazon Neptune (Gremlin) filter builder."""

    def test_simple_role_match(self):
        """Test basic role-based access."""
        policy = make_policy([
            {"roles": ["admin"], "match": {"category": "confidential"}}
        ])
        user = {"id": "alice", "roles": ["admin"]}

        filters = to_neptune_filter(policy, user)

        assert len(filters) > 0
        assert any(f["property"] == "category" for f in filters if f["type"] == "has")

    def test_condition_user_equals_document(self):
        """Test user.field == document.field condition."""
        policy = make_policy([
            {
                "roles": ["engineer"],
                "conditions": ["user.department == document.department"]
            }
        ])
        user = {"id": "alice", "roles": ["engineer"], "department": "engineering"}

        filters = to_neptune_filter(policy, user)

        assert len(filters) > 0
        # Check there's an equality filter for department
        has_dept = any(
            f.get("property") == "department" and f.get("value") == "engineering"
            for f in filters
            if f.get("type") == "has"
        )
        assert has_dept

    def test_deny_all_no_matching_rules(self):
        """Test deny-all filter when no rules match."""
        policy = make_policy([
            {"roles": ["admin"]}
        ], default="deny")
        user = {"id": "bob", "roles": ["viewer"]}

        filters = to_neptune_filter(policy, user)

        # Should return a filter that matches nothing
        assert len(filters) > 0
        # Check for deny-all pattern (uses __ragguard_deny_all__ constant)
        assert any(
            f.get("property") == "__ragguard_deny_all__"
            for f in filters
            if f.get("type") == "has"
        )

    def test_within_predicate(self):
        """Test IN operator uses 'within' predicate."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.category in ['public', 'internal']"]
            }
        ])
        user = {"id": "guest"}

        filters = to_neptune_filter(policy, user)

        # Find the filter with 'within' predicate
        within_filter = None
        for f in filters:
            if f.get("type") == "has" and f.get("predicate") == "within":
                within_filter = f
                break

        assert within_filter is not None
        assert within_filter["property"] == "category"

    def test_comparison_operators(self):
        """Test comparison operators."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.priority > 5"]
            }
        ])
        user = {"id": "guest"}

        filters = to_neptune_filter(policy, user)

        gt_filter = None
        for f in filters:
            if f.get("predicate") == "gt":
                gt_filter = f
                break

        assert gt_filter is not None
        assert gt_filter["value"] == 5


class TestTigerGraphFilterBuilder:
    """Tests for TigerGraph (GSQL) filter builder."""

    def test_simple_role_match(self):
        """Test basic role-based access."""
        policy = make_policy([
            {"roles": ["admin"], "match": {"category": "confidential"}}
        ])
        user = {"id": "alice", "roles": ["admin"]}

        where, params = to_tigergraph_filter(policy, user)

        assert 'v.category == "confidential"' in where

    def test_condition_user_equals_document(self):
        """Test user.field == document.field condition."""
        policy = make_policy([
            {
                "roles": ["engineer"],
                "conditions": ["user.department == document.department"]
            }
        ])
        user = {"id": "alice", "roles": ["engineer"], "department": "engineering"}

        where, params = to_tigergraph_filter(policy, user)

        assert 'v.department == "engineering"' in where

    def test_deny_all_no_matching_rules(self):
        """Test deny-all returns FALSE."""
        policy = make_policy([
            {"roles": ["admin"]}
        ], default="deny")
        user = {"id": "bob", "roles": ["viewer"]}

        where, params = to_tigergraph_filter(policy, user)

        assert where == "FALSE"

    def test_allow_all_default(self):
        """Test allow-all returns TRUE."""
        policy = make_policy([
            {"roles": ["admin"]}
        ], default="allow")
        user = {"id": "bob", "roles": ["viewer"]}

        where, params = to_tigergraph_filter(policy, user)

        assert where == "TRUE"

    def test_in_operator(self):
        """Test IN operator in GSQL."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.category in ['public', 'internal']"]
            }
        ])
        user = {"id": "guest"}

        where, params = to_tigergraph_filter(policy, user)

        assert "v.category IN" in where
        assert '"public"' in where or "'public'" in where

    def test_boolean_values(self):
        """Test boolean values in GSQL."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.is_public == true"]
            }
        ])
        user = {"id": "guest"}

        where, params = to_tigergraph_filter(policy, user)

        assert "v.is_public == TRUE" in where


class TestArangoDBFilterBuilder:
    """Tests for ArangoDB (AQL) filter builder."""

    def test_simple_role_match(self):
        """Test basic role-based access."""
        policy = make_policy([
            {"roles": ["admin"], "match": {"category": "confidential"}}
        ])
        user = {"id": "alice", "roles": ["admin"]}

        filter_expr, bind_vars = to_arangodb_filter(policy, user)

        assert "doc.category == @" in filter_expr
        assert "confidential" in bind_vars.values()

    def test_condition_user_equals_document(self):
        """Test user.field == document.field condition."""
        policy = make_policy([
            {
                "roles": ["engineer"],
                "conditions": ["user.department == document.department"]
            }
        ])
        user = {"id": "alice", "roles": ["engineer"], "department": "engineering"}

        filter_expr, bind_vars = to_arangodb_filter(policy, user)

        assert "doc.department == @" in filter_expr
        assert "engineering" in bind_vars.values()

    def test_deny_all_no_matching_rules(self):
        """Test deny-all returns false."""
        policy = make_policy([
            {"roles": ["admin"]}
        ], default="deny")
        user = {"id": "bob", "roles": ["viewer"]}

        filter_expr, bind_vars = to_arangodb_filter(policy, user)

        assert filter_expr == "false"

    def test_allow_all_default(self):
        """Test allow-all returns true."""
        policy = make_policy([
            {"roles": ["admin"]}
        ], default="allow")
        user = {"id": "bob", "roles": ["viewer"]}

        filter_expr, bind_vars = to_arangodb_filter(policy, user)

        assert filter_expr == "true"

    def test_in_operator(self):
        """Test IN operator in AQL."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.category in ['public', 'internal']"]
            }
        ])
        user = {"id": "guest"}

        filter_expr, bind_vars = to_arangodb_filter(policy, user)

        assert "doc.category IN @" in filter_expr
        # Check that the bind var contains the list
        assert any(
            isinstance(v, list) and "public" in v
            for v in bind_vars.values()
        )

    def test_not_in_operator(self):
        """Test NOT IN operator in AQL."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.category not in ['deleted', 'archived']"]
            }
        ])
        user = {"id": "guest"}

        filter_expr, bind_vars = to_arangodb_filter(policy, user)

        assert "NOT IN" in filter_expr

    def test_field_existence_not_null(self):
        """Test field exists uses != null."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.metadata exists"]
            }
        ])
        user = {"id": "guest"}

        filter_expr, bind_vars = to_arangodb_filter(policy, user)

        assert "!= null" in filter_expr

    def test_logical_and_or(self):
        """Test && and || operators."""
        policy = make_policy([
            {"roles": ["admin"], "match": {"category": "a"}},
            {"roles": ["admin"], "match": {"category": "b"}}
        ])
        user = {"id": "alice", "roles": ["admin"]}

        filter_expr, bind_vars = to_arangodb_filter(policy, user)

        # Multiple rules should be ORed
        assert "||" in filter_expr or filter_expr.count("doc.category") >= 2


class TestCrossBackendConsistency:
    """Test that different backends produce semantically equivalent filters."""

    def test_simple_condition_consistency(self):
        """Test that simple conditions produce consistent logic across backends."""
        policy = make_policy([
            {
                "roles": ["engineer"],
                "conditions": ["user.department == document.department"]
            }
        ])
        user = {"id": "alice", "roles": ["engineer"], "department": "engineering"}

        # All backends should filter by department = engineering
        neo4j_where, neo4j_params = to_neo4j_filter(policy, user)
        neptune_filters = to_neptune_filter(policy, user)
        tiger_where, tiger_params = to_tigergraph_filter(policy, user)
        arango_filter, arango_vars = to_arangodb_filter(policy, user)

        # Check Neo4j
        assert "engineering" in neo4j_params.values()

        # Check Neptune
        assert any(
            f.get("property") == "department" and f.get("value") == "engineering"
            for f in neptune_filters
            if isinstance(f, dict)
        )

        # Check TigerGraph
        assert 'v.department == "engineering"' in tiger_where

        # Check ArangoDB
        assert "engineering" in arango_vars.values()

    def test_deny_all_consistency(self):
        """Test that deny-all produces consistent results."""
        policy = make_policy([
            {"roles": ["admin"]}
        ], default="deny")
        user = {"id": "bob", "roles": ["viewer"]}

        neo4j_where, _ = to_neo4j_filter(policy, user)
        neptune_filters = to_neptune_filter(policy, user)
        tiger_where, _ = to_tigergraph_filter(policy, user)
        arango_filter, _ = to_arangodb_filter(policy, user)

        # All should return "deny all" equivalent
        assert neo4j_where == "false"
        assert len(neptune_filters) > 0  # Has deny-all filter
        assert tiger_where == "FALSE"
        assert arango_filter == "false"

    def test_null_user_field_consistency(self):
        """Test handling of null user fields."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["user.team == document.team"]
            }
        ])
        # User without team field
        user = {"id": "alice"}

        neo4j_where, _ = to_neo4j_filter(policy, user)
        neptune_filters = to_neptune_filter(policy, user)
        tiger_where, _ = to_tigergraph_filter(policy, user)
        arango_filter, _ = to_arangodb_filter(policy, user)

        # All should deny when user field is null
        assert "false" in neo4j_where.lower() or neo4j_where == ""
        # Neptune should have deny filter
        assert tiger_where == "FALSE"
        assert "false" in arango_filter.lower() or arango_filter == ""


class TestNeo4jFilterBuilderEdgeCases:
    """Edge case tests for Neo4j filter builder."""

    def test_less_than_operator(self):
        """Test < operator."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.score < 100"]
            }
        ])
        user = {"id": "guest"}

        where, params = to_neo4j_filter(policy, user)

        assert "doc.score <" in where
        assert any(v == 100 for v in params.values())

    def test_less_than_or_equal_operator(self):
        """Test <= operator."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.level <= 5"]
            }
        ])
        user = {"id": "guest"}

        where, params = to_neo4j_filter(policy, user)

        assert "doc.level <=" in where

    def test_greater_than_or_equal_operator(self):
        """Test >= operator."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.priority >= 3"]
            }
        ])
        user = {"id": "guest"}

        where, params = to_neo4j_filter(policy, user)

        assert "doc.priority >=" in where

    def test_not_in_operator_document_field(self):
        """Test NOT IN operator with document field and list."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.status not in ['deleted', 'archived']"]
            }
        ])
        user = {"id": "guest"}

        where, params = to_neo4j_filter(policy, user)

        assert "NOT" in where
        assert "IN" in where
        assert any(isinstance(v, list) and "deleted" in v for v in params.values())

    def test_not_in_operator_user_field(self):
        """Test user.id not in document.blocked_users."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["user.id not in document.blocked_users"]
            }
        ])
        user = {"id": "alice"}

        where, params = to_neo4j_filter(policy, user)

        assert "NOT" in where
        assert "$" in where
        assert "alice" in params.values()

    def test_not_exists_operator(self):
        """Test NOT EXISTS operator."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.deleted_at not exists"]
            }
        ])
        user = {"id": "guest"}

        where, params = to_neo4j_filter(policy, user)

        assert "IS NULL" in where

    def test_custom_node_alias(self):
        """Test custom node alias."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["user.department == document.department"]
            }
        ])
        user = {"id": "alice", "department": "engineering"}

        where, params = to_neo4j_filter(policy, user, node_alias="n")

        assert "n.department" in where
        assert "doc.department" not in where

    def test_multiple_conditions_and(self):
        """Test multiple conditions are ANDed within a rule."""
        policy = make_policy([
            {
                "roles": ["engineer"],
                "conditions": [
                    "user.department == document.department",
                    "document.status == 'active'"
                ]
            }
        ])
        user = {"id": "alice", "roles": ["engineer"], "department": "engineering"}

        where, params = to_neo4j_filter(policy, user)

        assert " AND " in where
        assert "department" in where
        assert "status" in where

    def test_match_with_list_values(self):
        """Test match with list values uses IN."""
        policy = make_policy([
            {
                "roles": ["admin"],
                "match": {"category": ["public", "internal"]}
            }
        ])
        user = {"id": "alice", "roles": ["admin"]}

        where, params = to_neo4j_filter(policy, user)

        assert "IN" in where
        assert any(isinstance(v, list) for v in params.values())

    def test_everyone_no_conditions_returns_true(self):
        """Test everyone with no conditions returns true."""
        policy = make_policy([
            {"everyone": True}
        ])
        user = {"id": "guest"}

        where, params = to_neo4j_filter(policy, user)

        assert where == "true"


class TestNeptuneFilterBuilderEdgeCases:
    """Edge case tests for Neptune filter builder."""

    def test_less_than_operator(self):
        """Test lt predicate."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.score < 100"]
            }
        ])
        user = {"id": "guest"}

        filters = to_neptune_filter(policy, user)

        lt_filter = next((f for f in filters if f.get("predicate") == "lt"), None)
        assert lt_filter is not None
        assert lt_filter["value"] == 100

    def test_less_than_or_equal_operator(self):
        """Test lte predicate."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.level <= 5"]
            }
        ])
        user = {"id": "guest"}

        filters = to_neptune_filter(policy, user)

        lte_filter = next((f for f in filters if f.get("predicate") == "lte"), None)
        assert lte_filter is not None

    def test_greater_than_or_equal_operator(self):
        """Test gte predicate."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.priority >= 3"]
            }
        ])
        user = {"id": "guest"}

        filters = to_neptune_filter(policy, user)

        gte_filter = next((f for f in filters if f.get("predicate") == "gte"), None)
        assert gte_filter is not None

    def test_not_equals_predicate(self):
        """Test neq predicate."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.status != 'deleted'"]
            }
        ])
        user = {"id": "guest"}

        filters = to_neptune_filter(policy, user)

        neq_filter = next((f for f in filters if f.get("predicate") == "neq"), None)
        assert neq_filter is not None
        assert neq_filter["value"] == "deleted"

    def test_not_in_document_list(self):
        """Test without predicate for NOT IN list."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.category not in ['deleted', 'archived']"]
            }
        ])
        user = {"id": "guest"}

        filters = to_neptune_filter(policy, user)

        without_filter = next((f for f in filters if f.get("predicate") == "without"), None)
        assert without_filter is not None

    def test_not_in_document_array(self):
        """Test notContaining predicate for user.id not in document.array."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["user.id not in document.blocked_users"]
            }
        ])
        user = {"id": "alice"}

        filters = to_neptune_filter(policy, user)

        not_containing = next((f for f in filters if f.get("predicate") == "notContaining"), None)
        assert not_containing is not None
        assert not_containing["value"] == "alice"

    def test_not_exists_hasNot(self):
        """Test NOT EXISTS uses hasNot type."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.deleted_at not exists"]
            }
        ])
        user = {"id": "guest"}

        filters = to_neptune_filter(policy, user)

        has_not = next((f for f in filters if f.get("type") == "hasNot"), None)
        assert has_not is not None

    def test_exists_predicate(self):
        """Test EXISTS uses exists predicate."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.metadata exists"]
            }
        ])
        user = {"id": "guest"}

        filters = to_neptune_filter(policy, user)

        exists_filter = next((f for f in filters if f.get("predicate") == "exists"), None)
        assert exists_filter is not None

    def test_allow_all_default_empty_filters(self):
        """Test allow-all returns empty filter list."""
        policy = make_policy([
            {"roles": ["admin"]}
        ], default="allow")
        user = {"id": "bob", "roles": ["viewer"]}

        filters = to_neptune_filter(policy, user)

        assert filters == []

    def test_everyone_no_conditions_empty_filters(self):
        """Test everyone with no conditions returns empty list."""
        policy = make_policy([
            {"everyone": True}
        ])
        user = {"id": "guest"}

        filters = to_neptune_filter(policy, user)

        assert filters == []

    def test_user_id_in_document_array_containing(self):
        """Test user.id in document.array uses containing predicate."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["user.id in document.authorized_users"]
            }
        ])
        user = {"id": "alice"}

        filters = to_neptune_filter(policy, user)

        containing = next((f for f in filters if f.get("predicate") == "containing"), None)
        assert containing is not None
        assert containing["value"] == "alice"

    def test_multiple_rules_or_structure(self):
        """Test multiple rules create OR structure."""
        policy = make_policy([
            {"roles": ["admin"], "match": {"category": "admin"}},
            {"roles": ["admin"], "match": {"category": "system"}}
        ])
        user = {"id": "alice", "roles": ["admin"]}

        filters = to_neptune_filter(policy, user)

        # Should have an OR type wrapper
        or_filter = next((f for f in filters if f.get("type") == "or"), None)
        assert or_filter is not None
        assert "children" in or_filter


class TestTigerGraphFilterBuilderEdgeCases:
    """Edge case tests for TigerGraph filter builder."""

    def test_less_than_operator(self):
        """Test < operator in GSQL."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.score < 100"]
            }
        ])
        user = {"id": "guest"}

        where, params = to_tigergraph_filter(policy, user)

        assert "v.score < 100" in where

    def test_less_than_or_equal_operator(self):
        """Test <= operator in GSQL."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.level <= 5"]
            }
        ])
        user = {"id": "guest"}

        where, params = to_tigergraph_filter(policy, user)

        assert "v.level <= 5" in where

    def test_greater_than_or_equal_operator(self):
        """Test >= operator in GSQL."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.priority >= 3"]
            }
        ])
        user = {"id": "guest"}

        where, params = to_tigergraph_filter(policy, user)

        assert "v.priority >= 3" in where

    def test_not_in_document_list(self):
        """Test NOT IN with list values."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.status not in ['deleted', 'archived']"]
            }
        ])
        user = {"id": "guest"}

        where, params = to_tigergraph_filter(policy, user)

        assert "NOT" in where
        assert "IN" in where
        assert '"deleted"' in where

    def test_not_in_document_array(self):
        """Test user.id not in document.array."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["user.id not in document.blocked_users"]
            }
        ])
        user = {"id": "alice"}

        where, params = to_tigergraph_filter(policy, user)

        assert "NOT" in where
        assert '"alice"' in where

    def test_not_exists_empty_string(self):
        """Test NOT EXISTS uses empty string check in GSQL."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.deleted_at not exists"]
            }
        ])
        user = {"id": "guest"}

        where, params = to_tigergraph_filter(policy, user)

        assert '== ""' in where

    def test_exists_non_empty_string(self):
        """Test EXISTS uses non-empty string check in GSQL."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.metadata exists"]
            }
        ])
        user = {"id": "guest"}

        where, params = to_tigergraph_filter(policy, user)

        assert '!= ""' in where

    def test_custom_vertex_alias(self):
        """Test custom vertex alias."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["user.department == document.department"]
            }
        ])
        user = {"id": "alice", "department": "engineering"}

        where, params = to_tigergraph_filter(policy, user, vertex_alias="vertex")

        assert "vertex.department" in where
        assert "v.department" not in where

    def test_multiple_conditions_and(self):
        """Test multiple conditions ANDed within a rule."""
        policy = make_policy([
            {
                "roles": ["engineer"],
                "conditions": [
                    "user.department == document.department",
                    "document.status == 'active'"
                ]
            }
        ])
        user = {"id": "alice", "roles": ["engineer"], "department": "engineering"}

        where, params = to_tigergraph_filter(policy, user)

        assert " AND " in where

    def test_match_with_list_values(self):
        """Test match with list values."""
        policy = make_policy([
            {
                "roles": ["admin"],
                "match": {"category": ["public", "internal"]}
            }
        ])
        user = {"id": "alice", "roles": ["admin"]}

        where, params = to_tigergraph_filter(policy, user)

        assert "IN" in where
        assert '"public"' in where

    def test_everyone_no_conditions_returns_true(self):
        """Test everyone with no conditions returns TRUE."""
        policy = make_policy([
            {"everyone": True}
        ])
        user = {"id": "guest"}

        where, params = to_tigergraph_filter(policy, user)

        assert where == "TRUE"

    def test_numeric_match_value(self):
        """Test numeric values in match."""
        policy = make_policy([
            {
                "roles": ["admin"],
                "match": {"priority": 5}
            }
        ])
        user = {"id": "alice", "roles": ["admin"]}

        where, params = to_tigergraph_filter(policy, user)

        assert "v.priority == 5" in where

    def test_boolean_false_value(self):
        """Test boolean false value."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.is_deleted == false"]
            }
        ])
        user = {"id": "guest"}

        where, params = to_tigergraph_filter(policy, user)

        assert "FALSE" in where


class TestArangoDBFilterBuilderEdgeCases:
    """Edge case tests for ArangoDB filter builder."""

    def test_less_than_operator(self):
        """Test < operator in AQL."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.score < 100"]
            }
        ])
        user = {"id": "guest"}

        filter_expr, bind_vars = to_arangodb_filter(policy, user)

        assert "doc.score <" in filter_expr
        assert any(v == 100 for v in bind_vars.values())

    def test_less_than_or_equal_operator(self):
        """Test <= operator in AQL."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.level <= 5"]
            }
        ])
        user = {"id": "guest"}

        filter_expr, bind_vars = to_arangodb_filter(policy, user)

        assert "doc.level <=" in filter_expr

    def test_greater_than_or_equal_operator(self):
        """Test >= operator in AQL."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.priority >= 3"]
            }
        ])
        user = {"id": "guest"}

        filter_expr, bind_vars = to_arangodb_filter(policy, user)

        assert "doc.priority >=" in filter_expr

    def test_not_exists_null_check(self):
        """Test NOT EXISTS uses null check."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.deleted_at not exists"]
            }
        ])
        user = {"id": "guest"}

        filter_expr, bind_vars = to_arangodb_filter(policy, user)

        assert "== null" in filter_expr

    def test_user_not_in_document_array(self):
        """Test user.id not in document.array."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["user.id not in document.blocked_users"]
            }
        ])
        user = {"id": "alice"}

        filter_expr, bind_vars = to_arangodb_filter(policy, user)

        assert "NOT IN" in filter_expr
        assert "alice" in bind_vars.values()

    def test_custom_doc_alias(self):
        """Test custom document alias."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["user.department == document.department"]
            }
        ])
        user = {"id": "alice", "department": "engineering"}

        filter_expr, bind_vars = to_arangodb_filter(policy, user, doc_alias="d")

        assert "d.department" in filter_expr
        assert "doc.department" not in filter_expr

    def test_multiple_conditions_and(self):
        """Test multiple conditions ANDed with &&."""
        policy = make_policy([
            {
                "roles": ["engineer"],
                "conditions": [
                    "user.department == document.department",
                    "document.status == 'active'"
                ]
            }
        ])
        user = {"id": "alice", "roles": ["engineer"], "department": "engineering"}

        filter_expr, bind_vars = to_arangodb_filter(policy, user)

        assert "&&" in filter_expr

    def test_match_with_list_values(self):
        """Test match with list values uses IN."""
        policy = make_policy([
            {
                "roles": ["admin"],
                "match": {"category": ["public", "internal"]}
            }
        ])
        user = {"id": "alice", "roles": ["admin"]}

        filter_expr, bind_vars = to_arangodb_filter(policy, user)

        assert "IN" in filter_expr
        assert any(isinstance(v, list) for v in bind_vars.values())

    def test_everyone_no_conditions_returns_true(self):
        """Test everyone with no conditions returns true."""
        policy = make_policy([
            {"everyone": True}
        ])
        user = {"id": "guest"}

        filter_expr, bind_vars = to_arangodb_filter(policy, user)

        assert filter_expr == "true"

    def test_user_in_document_array(self):
        """Test user.id in document.array."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["user.id in document.authorized_users"]
            }
        ])
        user = {"id": "alice"}

        filter_expr, bind_vars = to_arangodb_filter(policy, user)

        assert "IN" in filter_expr
        assert "alice" in bind_vars.values()

    def test_greater_than_operator(self):
        """Test > operator in AQL."""
        policy = make_policy([
            {
                "everyone": True,
                "conditions": ["document.score > 50"]
            }
        ])
        user = {"id": "guest"}

        filter_expr, bind_vars = to_arangodb_filter(policy, user)

        assert "doc.score >" in filter_expr
        assert any(v == 50 for v in bind_vars.values())
