"""Extended tests for graph database retrievers to improve coverage."""

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
# TigerGraph Extended Tests
# ============================================================

class TestTigerGraphExecuteGraphQuery:
    """Tests for TigerGraph _execute_graph_query method."""

    def test_run_query_without_permission_filter(self):
        """Test executing RUN query with no permission filter."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)
            mock_conn = MagicMock()
            retriever.connection = mock_conn
            retriever.vertex_type = "Document"
            retriever.vertex_alias = "v"

            mock_conn.runInstalledQuery.return_value = [{"id": "v1"}, {"id": "v2"}]

            results = retriever._execute_graph_query(
                query="RUN getDocuments()",
                permission_filter=("TRUE", {}),
                limit=10
            )

            assert len(results) == 2
            mock_conn.runInstalledQuery.assert_called_once()

    def test_run_query_with_permission_filter_and_post_filter(self):
        """Test executing RUN query with permission filter requiring post-filtering."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)
            mock_conn = MagicMock()
            retriever.connection = mock_conn
            retriever.vertex_type = "Document"
            retriever.vertex_alias = "v"

            mock_conn.runInstalledQuery.return_value = [
                {"id": "v1", "department": "eng"},
                {"id": "v2", "department": "sales"}
            ]

            # Mock _post_filter_results
            retriever._post_filter_results = Mock(return_value=[{"id": "v1", "department": "eng"}])

            results = retriever._execute_graph_query(
                query="RUN getDocuments()",
                permission_filter=("v.department == 'eng'", {}),
                limit=10
            )

            assert results == [{"id": "v1", "department": "eng"}]
            retriever._post_filter_results.assert_called_once()

    def test_select_query_with_where_clause(self):
        """Test SELECT query with existing WHERE clause."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)
            mock_conn = MagicMock()
            retriever.connection = mock_conn
            retriever.vertex_type = "Document"
            retriever.vertex_alias = "v"

            mock_conn.gsql.return_value = [{"id": "v1"}]

            results = retriever._execute_graph_query(
                query="SELECT v FROM Document:v WHERE v.category == 'eng'",
                permission_filter=("v.department == 'sales'", {}),
                limit=10
            )

            # Check that WHERE clause was modified
            call_args = mock_conn.gsql.call_args[0][0]
            assert "(v.department == 'sales') AND" in call_args
            assert "LIMIT 10" in call_args

    def test_select_query_with_order_clause(self):
        """Test SELECT query with ORDER clause but no WHERE."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)
            mock_conn = MagicMock()
            retriever.connection = mock_conn
            retriever.vertex_type = "Document"
            retriever.vertex_alias = "v"

            mock_conn.gsql.return_value = [{"id": "v1"}]

            results = retriever._execute_graph_query(
                query="SELECT v FROM Document:v ORDER BY v.score",
                permission_filter=("v.department == 'eng'", {}),
                limit=10
            )

            # Check that WHERE was inserted before ORDER
            call_args = mock_conn.gsql.call_args[0][0]
            assert "WHERE v.department == 'eng'" in call_args

    def test_select_query_with_limit_clause(self):
        """Test SELECT query with LIMIT clause."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)
            mock_conn = MagicMock()
            retriever.connection = mock_conn
            retriever.vertex_type = "Document"
            retriever.vertex_alias = "v"

            mock_conn.gsql.return_value = [{"id": "v1"}]

            results = retriever._execute_graph_query(
                query="SELECT v FROM Document:v LIMIT 5",
                permission_filter=("v.department == 'eng'", {}),
                limit=10
            )

            # Check that WHERE was inserted before LIMIT
            call_args = mock_conn.gsql.call_args[0][0]
            assert "WHERE v.department == 'eng'" in call_args

    def test_query_without_select(self):
        """Test query without SELECT keyword."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)
            mock_conn = MagicMock()
            retriever.connection = mock_conn
            retriever.vertex_type = "Document"
            retriever.vertex_alias = "v"

            mock_conn.gsql.return_value = [{"id": "v1"}]

            # Non-SELECT query, permission filter still applied
            results = retriever._execute_graph_query(
                query="INTERPRET QUERY () { PRINT 'hello'; }",
                permission_filter=("v.active == TRUE", {}),
                limit=10
            )

            # Query should be passed through with LIMIT appended
            call_args = mock_conn.gsql.call_args[0][0]
            assert "LIMIT 10" in call_args

    def test_query_without_permission_filter(self):
        """Test query with empty permission filter."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)
            mock_conn = MagicMock()
            retriever.connection = mock_conn
            retriever.vertex_type = "Document"
            retriever.vertex_alias = "v"

            mock_conn.gsql.return_value = [{"id": "v1"}]

            results = retriever._execute_graph_query(
                query="SELECT v FROM Document:v",
                permission_filter=("", {}),
                limit=10
            )

            call_args = mock_conn.gsql.call_args[0][0]
            assert "LIMIT 10" in call_args


class TestTigerGraphPropertySearch:
    """Tests for TigerGraph _execute_property_search method."""

    def test_property_search_with_string_value(self):
        """Test property search with string values."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)
            mock_conn = MagicMock()
            retriever.connection = mock_conn
            retriever.vertex_type = "Document"
            retriever.vertex_alias = "v"

            # REST API success
            mock_conn.getVertices.return_value = [
                {"v_id": "v1", "attributes": {"category": "engineering", "department": "eng"}}
            ]

            retriever._matches_properties = Mock(return_value=True)
            retriever._matches_permission_filter = Mock(return_value=True)

            results = retriever._execute_property_search(
                properties={"category": "engineering"},
                permission_filter=("v.department == 'eng'", {}),
                limit=10
            )

            assert len(results) == 1
            mock_conn.getVertices.assert_called_once()

    def test_property_search_with_bool_value(self):
        """Test property search with boolean values."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)
            mock_conn = MagicMock()
            retriever.connection = mock_conn
            retriever.vertex_type = "Document"
            retriever.vertex_alias = "v"

            mock_conn.getVertices.return_value = [
                {"v_id": "v1", "attributes": {"active": True}}
            ]

            retriever._matches_properties = Mock(return_value=True)
            retriever._matches_permission_filter = Mock(return_value=True)

            results = retriever._execute_property_search(
                properties={"active": True},
                permission_filter=("TRUE", {}),
                limit=10
            )

            assert len(results) == 1

    def test_property_search_with_int_value(self):
        """Test property search with integer values."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)
            mock_conn = MagicMock()
            retriever.connection = mock_conn
            retriever.vertex_type = "Document"
            retriever.vertex_alias = "v"

            mock_conn.getVertices.return_value = [
                {"v_id": "v1", "attributes": {"priority": 5}}
            ]

            retriever._matches_properties = Mock(return_value=True)
            retriever._matches_permission_filter = Mock(return_value=True)

            results = retriever._execute_property_search(
                properties={"priority": 5},
                permission_filter=("TRUE", {}),
                limit=10
            )

            assert len(results) == 1

    def test_property_search_with_list_value(self):
        """Test property search with list values."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)
            mock_conn = MagicMock()
            retriever.connection = mock_conn
            retriever.vertex_type = "Document"
            retriever.vertex_alias = "v"

            mock_conn.getVertices.return_value = [
                {"v_id": "v1", "attributes": {"category": "engineering"}}
            ]

            retriever._matches_properties = Mock(return_value=True)
            retriever._matches_permission_filter = Mock(return_value=True)

            results = retriever._execute_property_search(
                properties={"category": ["engineering", "sales"]},
                permission_filter=("TRUE", {}),
                limit=10
            )

            assert len(results) == 1

    def test_property_search_rest_api_fallback(self):
        """Test property search falls back to GSQL when REST API fails."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)
            mock_conn = MagicMock()
            retriever.connection = mock_conn
            retriever.vertex_type = "Document"
            retriever.vertex_alias = "v"

            # REST API fails
            mock_conn.getVertices.side_effect = Exception("REST API error")
            # GSQL fallback
            mock_conn.gsql.return_value = [{"id": "v1"}]

            results = retriever._execute_property_search(
                properties={"category": "engineering"},
                permission_filter=("TRUE", {}),
                limit=10
            )

            mock_conn.gsql.assert_called_once()

    def test_property_search_filter_mismatch(self):
        """Test property search filters out non-matching vertices."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)
            mock_conn = MagicMock()
            retriever.connection = mock_conn
            retriever.vertex_type = "Document"
            retriever.vertex_alias = "v"

            mock_conn.getVertices.return_value = [
                {"v_id": "v1", "attributes": {"category": "sales"}},
                {"v_id": "v2", "attributes": {"category": "engineering"}}
            ]

            # Only match engineering category
            def check_props(v, props):
                attrs = v.get("attributes", v)
                return attrs.get("category") == "engineering"

            retriever._matches_properties = check_props
            retriever._matches_permission_filter = Mock(return_value=True)

            results = retriever._execute_property_search(
                properties={"category": "engineering"},
                permission_filter=("TRUE", {}),
                limit=10
            )

            assert len(results) == 1
            assert results[0]["v_id"] == "v2"


class TestTigerGraphMatchesProperties:
    """Tests for TigerGraph _matches_properties method."""

    def test_matches_with_attributes_wrapper(self):
        """Test matching with attributes wrapper."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)

            vertex = {"v_id": "v1", "attributes": {"category": "eng", "priority": 5}}

            assert retriever._matches_properties(vertex, {"category": "eng"})
            assert retriever._matches_properties(vertex, {"priority": 5})
            assert not retriever._matches_properties(vertex, {"category": "sales"})

    def test_matches_without_attributes_wrapper(self):
        """Test matching without attributes wrapper."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)

            vertex = {"category": "eng", "priority": 5}

            assert retriever._matches_properties(vertex, {"category": "eng"})
            assert not retriever._matches_properties(vertex, {"category": "sales"})

    def test_matches_with_list_value(self):
        """Test matching with list property values."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)

            vertex = {"category": "eng"}

            # Value in list
            assert retriever._matches_properties(vertex, {"category": ["eng", "sales"]})
            # Value not in list
            assert not retriever._matches_properties(vertex, {"category": ["sales", "hr"]})


class TestTigerGraphMatchesPermissionFilter:
    """Tests for TigerGraph _matches_permission_filter method."""

    def test_matches_empty_filter(self):
        """Test matching with empty permission filter."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)

            vertex = {"category": "eng"}

            assert retriever._matches_permission_filter(vertex, ("", {}))
            assert retriever._matches_permission_filter(vertex, ("TRUE", {}))

    def test_matches_false_filter(self):
        """Test matching with FALSE permission filter."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)

            vertex = {"category": "eng"}

            assert not retriever._matches_permission_filter(vertex, ("FALSE", {}))

    def test_matches_complex_filter_default(self):
        """Test matching with complex filter defaults to False (deny) for security."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)

            vertex = {"category": "eng", "attributes": {"department": "engineering"}}

            # SECURITY: Complex filters default to False (deny by default)
            # This follows fail-secure principle: if we can't evaluate, deny access
            assert not retriever._matches_permission_filter(vertex, ("v.department == 'eng'", {}))


class TestTigerGraphPostFilterResults:
    """Tests for TigerGraph _post_filter_results method."""

    def test_post_filter_results(self):
        """Test post-filtering results."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)

            results = [
                {"id": "v1", "department": "eng"},
                {"id": "v2", "department": "sales"}
            ]

            # Mock permission filter to accept eng only
            retriever._matches_permission_filter = lambda v, f: v.get("department") == "eng"

            filtered = retriever._post_filter_results(results, ("v.department == 'eng'", {}))

            assert len(filtered) == 1
            assert filtered[0]["id"] == "v1"


class TestTigerGraphTraversal:
    """Tests for TigerGraph _execute_traversal method."""

    def test_traversal_outgoing(self):
        """Test outgoing traversal."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)
            mock_conn = MagicMock()
            retriever.connection = mock_conn
            retriever.vertex_type = "Document"
            retriever.vertex_alias = "v"
            retriever._policy = make_policy([{"everyone": True}])
            retriever.collection = "Document"

            retriever._build_permission_clause = Mock(return_value=("TRUE", {}))
            mock_conn.gsql.return_value = [{"id": "v2"}]

            results = retriever._execute_traversal(
                start_node_id="v1",
                relationship_type="RELATES_TO",
                user={"id": "alice"},
                direction="outgoing",
                depth=1,
                limit=10
            )

            call_args = mock_conn.gsql.call_args[0][0]
            assert "-(RELATES_TO)->" in call_args

    def test_traversal_incoming(self):
        """Test incoming traversal."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)
            mock_conn = MagicMock()
            retriever.connection = mock_conn
            retriever.vertex_type = "Document"
            retriever.vertex_alias = "v"
            retriever._policy = make_policy([{"everyone": True}])
            retriever.collection = "Document"

            retriever._build_permission_clause = Mock(return_value=("TRUE", {}))
            mock_conn.gsql.return_value = [{"id": "v2"}]

            results = retriever._execute_traversal(
                start_node_id="v1",
                relationship_type="RELATES_TO",
                user={"id": "alice"},
                direction="incoming",
                depth=1,
                limit=10
            )

            call_args = mock_conn.gsql.call_args[0][0]
            assert "<-(RELATES_TO)-" in call_args

    def test_traversal_both_directions(self):
        """Test bidirectional traversal."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)
            mock_conn = MagicMock()
            retriever.connection = mock_conn
            retriever.vertex_type = "Document"
            retriever.vertex_alias = "v"
            retriever._policy = make_policy([{"everyone": True}])
            retriever.collection = "Document"

            retriever._build_permission_clause = Mock(return_value=("TRUE", {}))
            mock_conn.gsql.return_value = [{"id": "v2"}]

            results = retriever._execute_traversal(
                start_node_id="v1",
                relationship_type="RELATES_TO",
                user={"id": "alice"},
                direction="both",
                depth=1,
                limit=10
            )

            call_args = mock_conn.gsql.call_args[0][0]
            assert "-(RELATES_TO)-" in call_args
            assert "->" not in call_args or "<-" not in call_args or "-(RELATES_TO)-" in call_args

    def test_traversal_multi_depth(self):
        """Test multi-depth traversal."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)
            mock_conn = MagicMock()
            retriever.connection = mock_conn
            retriever.vertex_type = "Document"
            retriever.vertex_alias = "v"
            retriever._policy = make_policy([{"everyone": True}])
            retriever.collection = "Document"

            retriever._build_permission_clause = Mock(return_value=("TRUE", {}))
            mock_conn.gsql.return_value = [{"id": "v2"}]

            results = retriever._execute_traversal(
                start_node_id="v1",
                relationship_type="RELATES_TO",
                user={"id": "alice"},
                direction="outgoing",
                depth=3,
                limit=10
            )

            call_args = mock_conn.gsql.call_args[0][0]
            assert "*1..3" in call_args

    def test_traversal_with_permission_filter(self):
        """Test traversal with permission filter."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)
            mock_conn = MagicMock()
            retriever.connection = mock_conn
            retriever.vertex_type = "Document"
            retriever.vertex_alias = "v"
            retriever._policy = make_policy([{"everyone": True, "conditions": ["document.active == True"]}])
            retriever.collection = "Document"

            retriever._build_permission_clause = Mock(return_value=("v.active == TRUE", {}))
            mock_conn.gsql.return_value = [{"id": "v2"}]

            results = retriever._execute_traversal(
                start_node_id="v1",
                relationship_type="RELATES_TO",
                user={"id": "alice"},
                direction="outgoing",
                depth=1,
                limit=10
            )

            call_args = mock_conn.gsql.call_args[0][0]
            assert "WHERE" in call_args


class TestTigerGraphHealthCheck:
    """Tests for TigerGraph _check_backend_health method."""

    def test_health_check_with_statistics(self):
        """Test health check with statistics available."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)
            mock_conn = MagicMock()
            retriever.connection = mock_conn
            retriever.vertex_type = "Document"
            retriever.graph_name = "TestGraph"

            mock_conn.getStatistics.return_value = {"vertexCount": 100}
            mock_conn.graphname = "TestGraph"

            health = retriever._check_backend_health()

            assert health["connection_alive"] is True
            assert health["vertex_type"] == "Document"
            assert health["statistics"] == {"vertexCount": 100}

    def test_health_check_fallback_to_echo(self):
        """Test health check falls back to echo when statistics fail."""
        from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever

        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)
            mock_conn = MagicMock()
            retriever.connection = mock_conn
            retriever.vertex_type = "Document"
            retriever.graph_name = None

            mock_conn.getStatistics.side_effect = Exception("Stats not available")
            mock_conn.echo.return_value = {"message": "Hello GSQL"}
            mock_conn.graphname = "DefaultGraph"

            health = retriever._check_backend_health()

            assert health["connection_alive"] is True
            assert health["graph_name"] == "DefaultGraph"


# ============================================================
# Neptune Extended Tests
# ============================================================

class TestNeptuneExecuteGraphQuery:
    """Tests for Neptune _execute_graph_query method."""

    def test_execute_query_with_post_filter(self):
        """Test executing Gremlin query with post-filtering."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)
            mock_client = MagicMock()
            retriever.gremlin_client = mock_client
            retriever.node_label = "Document"

            result_set = MagicMock()
            result_set.all.return_value.result.return_value = [
                {"id": "v1", "category": "eng"},
                {"id": "v2", "category": "sales"}
            ]
            mock_client.submit.return_value = result_set

            # Filter to only eng category
            retriever._matches_filters = Mock(side_effect=lambda v, f: v.get("category") == "eng")

            results = retriever._execute_graph_query(
                query="g.V().hasLabel('Document')",
                permission_filter=[{"type": "has", "property": "category", "predicate": "eq", "value": "eng"}],
                limit=10
            )

            assert len(results) == 1
            assert results[0]["id"] == "v1"

    def test_execute_query_without_filter(self):
        """Test executing query without permission filter."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)
            mock_client = MagicMock()
            retriever.gremlin_client = mock_client

            result_set = MagicMock()
            result_set.all.return_value.result.return_value = [{"id": "v1"}, {"id": "v2"}]
            mock_client.submit.return_value = result_set

            results = retriever._execute_graph_query(
                query="g.V().hasLabel('Document')",
                permission_filter=[],
                limit=10
            )

            assert len(results) == 2

    def test_execute_query_with_limit(self):
        """Test query respects limit."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)
            mock_client = MagicMock()
            retriever.gremlin_client = mock_client

            result_set = MagicMock()
            result_set.all.return_value.result.return_value = [{"id": f"v{i}"} for i in range(20)]
            mock_client.submit.return_value = result_set

            results = retriever._execute_graph_query(
                query="g.V()",
                permission_filter=[],
                limit=5
            )

            assert len(results) == 5


class TestNeptunePropertySearch:
    """Tests for Neptune _execute_property_search method."""

    def test_property_search_with_string(self):
        """Test property search with string values."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)
            mock_client = MagicMock()
            retriever.gremlin_client = mock_client
            retriever.node_label = "Document"

            result_set = MagicMock()
            result_set.all.return_value.result.return_value = [{"id": "v1"}]
            mock_client.submit.return_value = result_set

            # Mock the gremlin_python import inside the function
            mock_P = MagicMock()
            with patch.dict('sys.modules', {'gremlin_python': MagicMock(), 'gremlin_python.process': MagicMock(), 'gremlin_python.process.traversal': MagicMock(P=mock_P)}):
                results = retriever._execute_property_search(
                    properties={"category": "engineering"},
                    permission_filter=[],
                    limit=10
                )

            # Verify query structure
            call_args = mock_client.submit.call_args[0][0]
            assert ".has('category', 'engineering')" in call_args

    def test_property_search_with_int(self):
        """Test property search with integer values."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)
            mock_client = MagicMock()
            retriever.gremlin_client = mock_client
            retriever.node_label = "Document"

            result_set = MagicMock()
            result_set.all.return_value.result.return_value = [{"id": "v1"}]
            mock_client.submit.return_value = result_set

            mock_P = MagicMock()
            with patch.dict('sys.modules', {'gremlin_python': MagicMock(), 'gremlin_python.process': MagicMock(), 'gremlin_python.process.traversal': MagicMock(P=mock_P)}):
                results = retriever._execute_property_search(
                    properties={"priority": 5},
                    permission_filter=[],
                    limit=10
                )

            call_args = mock_client.submit.call_args[0][0]
            assert ".has('priority', 5)" in call_args

    def test_property_search_with_bool(self):
        """Test property search with boolean values."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)
            mock_client = MagicMock()
            retriever.gremlin_client = mock_client
            retriever.node_label = "Document"

            result_set = MagicMock()
            result_set.all.return_value.result.return_value = [{"id": "v1"}]
            mock_client.submit.return_value = result_set

            mock_P = MagicMock()
            with patch.dict('sys.modules', {'gremlin_python': MagicMock(), 'gremlin_python.process': MagicMock(), 'gremlin_python.process.traversal': MagicMock(P=mock_P)}):
                results = retriever._execute_property_search(
                    properties={"active": True},
                    permission_filter=[],
                    limit=10
                )

            call_args = mock_client.submit.call_args[0][0]
            # Python bools serialize to True/False
            assert ".has('active', True)" in call_args or ".has('active', true)" in call_args

    def test_property_search_with_list(self):
        """Test property search with list values."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)
            mock_client = MagicMock()
            retriever.gremlin_client = mock_client
            retriever.node_label = "Document"

            result_set = MagicMock()
            result_set.all.return_value.result.return_value = [{"id": "v1"}]
            mock_client.submit.return_value = result_set

            mock_P = MagicMock()
            with patch.dict('sys.modules', {'gremlin_python': MagicMock(), 'gremlin_python.process': MagicMock(), 'gremlin_python.process.traversal': MagicMock(P=mock_P)}):
                results = retriever._execute_property_search(
                    properties={"category": ["eng", "sales"]},
                    permission_filter=[],
                    limit=10
                )

            call_args = mock_client.submit.call_args[0][0]
            assert "within('eng', 'sales')" in call_args

    def test_property_search_with_permission_filters(self):
        """Test property search with permission filters."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)
            mock_client = MagicMock()
            retriever.gremlin_client = mock_client
            retriever.node_label = "Document"

            result_set = MagicMock()
            result_set.all.return_value.result.return_value = [{"id": "v1"}]
            mock_client.submit.return_value = result_set

            retriever._filter_to_gremlin_string = Mock(return_value=".has('department', 'eng')")

            mock_P = MagicMock()
            with patch.dict('sys.modules', {'gremlin_python': MagicMock(), 'gremlin_python.process': MagicMock(), 'gremlin_python.process.traversal': MagicMock(P=mock_P)}):
                results = retriever._execute_property_search(
                    properties={"category": "engineering"},
                    permission_filter=[{"type": "has", "property": "department", "predicate": "eq", "value": "eng"}],
                    limit=10
                )

            call_args = mock_client.submit.call_args[0][0]
            assert ".has('department', 'eng')" in call_args


class TestNeptuneFilterToGremlinString:
    """Tests for Neptune _filter_to_gremlin_string method."""

    def test_has_not_filter(self):
        """Test hasNot filter conversion."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)

            result = retriever._filter_to_gremlin_string({
                "type": "hasNot",
                "property": "deleted"
            })

            assert result == ".hasNot('deleted')"

    def test_eq_filter_string(self):
        """Test eq filter with string value."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)

            result = retriever._filter_to_gremlin_string({
                "type": "has",
                "property": "category",
                "predicate": "eq",
                "value": "engineering"
            })

            assert result == ".has('category', 'engineering')"

    def test_eq_filter_int(self):
        """Test eq filter with integer value."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)

            result = retriever._filter_to_gremlin_string({
                "type": "has",
                "property": "priority",
                "predicate": "eq",
                "value": 5
            })

            assert result == ".has('priority', 5)"

    def test_neq_filter(self):
        """Test neq filter conversion."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)

            result = retriever._filter_to_gremlin_string({
                "type": "has",
                "property": "status",
                "predicate": "neq",
                "value": "deleted"
            })

            assert result == ".has('status', neq('deleted'))"

    def test_within_filter(self):
        """Test within filter conversion."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)

            result = retriever._filter_to_gremlin_string({
                "type": "has",
                "property": "category",
                "predicate": "within",
                "value": ["eng", "sales"]
            })

            assert result == ".has('category', within('eng', 'sales'))"

    def test_without_filter(self):
        """Test without filter conversion."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)

            result = retriever._filter_to_gremlin_string({
                "type": "has",
                "property": "category",
                "predicate": "without",
                "value": ["archived", "deleted"]
            })

            assert result == ".has('category', without('archived', 'deleted'))"

    def test_comparison_filters(self):
        """Test comparison filter conversions."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)

            assert retriever._filter_to_gremlin_string({
                "type": "has", "property": "score", "predicate": "gt", "value": 5
            }) == ".has('score', gt(5))"

            assert retriever._filter_to_gremlin_string({
                "type": "has", "property": "score", "predicate": "gte", "value": 5
            }) == ".has('score', gte(5))"

            assert retriever._filter_to_gremlin_string({
                "type": "has", "property": "score", "predicate": "lt", "value": 10
            }) == ".has('score', lt(10))"

            assert retriever._filter_to_gremlin_string({
                "type": "has", "property": "score", "predicate": "lte", "value": 10
            }) == ".has('score', lte(10))"

    def test_exists_filter(self):
        """Test exists filter conversion."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)

            result = retriever._filter_to_gremlin_string({
                "type": "has",
                "property": "email",
                "predicate": "exists",
                "value": True
            })

            assert result == ".has('email')"

    def test_or_filter(self):
        """Test OR filter conversion."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)

            result = retriever._filter_to_gremlin_string({
                "type": "or",
                "children": [
                    [{"type": "has", "property": "category", "predicate": "eq", "value": "eng"}],
                    [{"type": "has", "property": "category", "predicate": "eq", "value": "sales"}]
                ]
            })

            assert ".or(" in result
            assert "has('category', 'eng')" in result
            assert "has('category', 'sales')" in result

    def test_unknown_type_returns_empty(self):
        """Test unknown filter type returns empty string."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)

            result = retriever._filter_to_gremlin_string({
                "type": "unknown",
                "property": "foo"
            })

            assert result == ""


class TestNeptuneTraversal:
    """Tests for Neptune _execute_traversal method."""

    def test_traversal_outgoing(self):
        """Test outgoing traversal."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)
            mock_client = MagicMock()
            retriever.gremlin_client = mock_client
            retriever.node_label = "Document"
            retriever._policy = make_policy([{"everyone": True}])
            retriever.collection = "Document"

            retriever._build_permission_clause = Mock(return_value=[])
            retriever._filter_to_gremlin_string = Mock(return_value="")

            result_set = MagicMock()
            result_set.all.return_value.result.return_value = [{"id": "v2"}]
            mock_client.submit.return_value = result_set

            results = retriever._execute_traversal(
                start_node_id="v1",
                relationship_type="RELATES_TO",
                user={"id": "alice"},
                direction="outgoing",
                depth=1,
                limit=10
            )

            call_args = mock_client.submit.call_args[0][0]
            assert ".out('RELATES_TO')" in call_args

    def test_traversal_incoming(self):
        """Test incoming traversal."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)
            mock_client = MagicMock()
            retriever.gremlin_client = mock_client
            retriever.node_label = "Document"

            retriever._build_permission_clause = Mock(return_value=[])
            retriever._filter_to_gremlin_string = Mock(return_value="")

            result_set = MagicMock()
            result_set.all.return_value.result.return_value = [{"id": "v2"}]
            mock_client.submit.return_value = result_set

            results = retriever._execute_traversal(
                start_node_id="v1",
                relationship_type="RELATES_TO",
                user={"id": "alice"},
                direction="incoming",
                depth=1,
                limit=10
            )

            call_args = mock_client.submit.call_args[0][0]
            assert ".in('RELATES_TO')" in call_args

    def test_traversal_both_directions(self):
        """Test bidirectional traversal."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)
            mock_client = MagicMock()
            retriever.gremlin_client = mock_client
            retriever.node_label = "Document"

            retriever._build_permission_clause = Mock(return_value=[])
            retriever._filter_to_gremlin_string = Mock(return_value="")

            result_set = MagicMock()
            result_set.all.return_value.result.return_value = [{"id": "v2"}]
            mock_client.submit.return_value = result_set

            results = retriever._execute_traversal(
                start_node_id="v1",
                relationship_type="RELATES_TO",
                user={"id": "alice"},
                direction="both",
                depth=1,
                limit=10
            )

            call_args = mock_client.submit.call_args[0][0]
            assert ".both('RELATES_TO')" in call_args

    def test_traversal_multi_depth(self):
        """Test multi-depth traversal."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)
            mock_client = MagicMock()
            retriever.gremlin_client = mock_client
            retriever.node_label = "Document"

            retriever._build_permission_clause = Mock(return_value=[])
            retriever._filter_to_gremlin_string = Mock(return_value="")

            result_set = MagicMock()
            result_set.all.return_value.result.return_value = [{"id": "v2"}]
            mock_client.submit.return_value = result_set

            results = retriever._execute_traversal(
                start_node_id="v1",
                relationship_type="RELATES_TO",
                user={"id": "alice"},
                direction="outgoing",
                depth=3,
                limit=10
            )

            call_args = mock_client.submit.call_args[0][0]
            assert ".repeat(" in call_args
            assert ".times(3)" in call_args


class TestNeptuneHealthCheckAndExit:
    """Tests for Neptune health check and context manager."""

    def test_health_check(self):
        """Test health check."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)
            mock_client = MagicMock()
            retriever.gremlin_client = mock_client
            retriever.node_label = "Document"

            result_set = MagicMock()
            result_set.all.return_value.result.return_value = [1]
            mock_client.submit.return_value = result_set

            health = retriever._check_backend_health()

            assert health["connection_alive"] is True
            assert health["can_query"] is True

    def test_exit_closes_client(self):
        """Test __exit__ closes the Gremlin client."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)
            mock_client = MagicMock()
            retriever.gremlin_client = mock_client

            # Mock superclass __exit__
            with patch.object(NeptuneSecureRetriever.__bases__[0], '__exit__', return_value=None):
                retriever.__exit__(None, None, None)

            mock_client.close.assert_called_once()

    def test_exit_handles_close_error(self):
        """Test __exit__ handles errors when closing client."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)
            mock_client = MagicMock()
            mock_client.close.side_effect = Exception("Close error")
            retriever.gremlin_client = mock_client

            # Mock superclass __exit__
            with patch.object(NeptuneSecureRetriever.__bases__[0], '__exit__', return_value=None):
                # Should not raise
                retriever.__exit__(None, None, None)


# ============================================================
# Neo4j Extended Tests
# ============================================================

class TestNeo4jGraphQuery:
    """Tests for Neo4j _execute_graph_query method."""

    def test_query_with_existing_where_clause(self):
        """Test query with existing WHERE clause."""
        from ragguard.retrievers.neo4j import Neo4jSecureRetriever

        with patch.object(Neo4jSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = Neo4jSecureRetriever.__new__(Neo4jSecureRetriever)
            mock_driver = MagicMock()
            retriever.driver = mock_driver
            retriever.database = None
            retriever.node_alias = "doc"

            session = MagicMock()
            result = MagicMock()
            result.__iter__ = Mock(return_value=iter([Mock(data=Mock(return_value={"id": "doc1"}))]))
            session.run.return_value = result
            mock_driver.session.return_value.__enter__ = Mock(return_value=session)
            mock_driver.session.return_value.__exit__ = Mock(return_value=False)

            results = retriever._execute_graph_query(
                query="MATCH (doc:Document) WHERE doc.active = true RETURN doc",
                permission_filter=("doc.department = $dept", {"dept": "eng"}),
                limit=10
            )

            call_args = session.run.call_args[0][0]
            assert "(doc.department = $dept) AND" in call_args
            assert "LIMIT 10" in call_args

    def test_query_with_return_but_no_where(self):
        """Test query with RETURN but no WHERE clause."""
        from ragguard.retrievers.neo4j import Neo4jSecureRetriever

        with patch.object(Neo4jSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = Neo4jSecureRetriever.__new__(Neo4jSecureRetriever)
            mock_driver = MagicMock()
            retriever.driver = mock_driver
            retriever.database = None
            retriever.node_alias = "doc"

            session = MagicMock()
            result = MagicMock()
            result.__iter__ = Mock(return_value=iter([Mock(data=Mock(return_value={"id": "doc1"}))]))
            session.run.return_value = result
            mock_driver.session.return_value.__enter__ = Mock(return_value=session)
            mock_driver.session.return_value.__exit__ = Mock(return_value=False)

            results = retriever._execute_graph_query(
                query="MATCH (doc:Document) RETURN doc",
                permission_filter=("doc.department = $dept", {"dept": "eng"}),
                limit=10
            )

            call_args = session.run.call_args[0][0]
            assert "WHERE doc.department = $dept" in call_args

    def test_query_without_where_or_return(self):
        """Test query without WHERE or RETURN appends WHERE."""
        from ragguard.retrievers.neo4j import Neo4jSecureRetriever

        with patch.object(Neo4jSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = Neo4jSecureRetriever.__new__(Neo4jSecureRetriever)
            mock_driver = MagicMock()
            retriever.driver = mock_driver
            retriever.database = None
            retriever.node_alias = "doc"

            session = MagicMock()
            result = MagicMock()
            result.__iter__ = Mock(return_value=iter([]))
            session.run.return_value = result
            mock_driver.session.return_value.__enter__ = Mock(return_value=session)
            mock_driver.session.return_value.__exit__ = Mock(return_value=False)

            results = retriever._execute_graph_query(
                query="MATCH (doc:Document)",
                permission_filter=("doc.department = $dept", {"dept": "eng"}),
                limit=10
            )

            call_args = session.run.call_args[0][0]
            assert "WHERE doc.department = $dept" in call_args

    def test_query_with_empty_permission_filter(self):
        """Test query with empty permission filter."""
        from ragguard.retrievers.neo4j import Neo4jSecureRetriever

        with patch.object(Neo4jSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = Neo4jSecureRetriever.__new__(Neo4jSecureRetriever)
            mock_driver = MagicMock()
            retriever.driver = mock_driver
            retriever.database = None
            retriever.node_alias = "doc"

            session = MagicMock()
            result = MagicMock()
            result.__iter__ = Mock(return_value=iter([Mock(data=Mock(return_value={"id": "doc1"}))]))
            session.run.return_value = result
            mock_driver.session.return_value.__enter__ = Mock(return_value=session)
            mock_driver.session.return_value.__exit__ = Mock(return_value=False)

            results = retriever._execute_graph_query(
                query="MATCH (doc:Document) RETURN doc",
                permission_filter=("", {}),
                limit=10
            )

            # Query should be passed through without permission filter injection
            call_args = session.run.call_args[0][0]
            assert "LIMIT 10" in call_args

    def test_query_with_existing_limit(self):
        """Test query with existing LIMIT clause."""
        from ragguard.retrievers.neo4j import Neo4jSecureRetriever

        with patch.object(Neo4jSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = Neo4jSecureRetriever.__new__(Neo4jSecureRetriever)
            mock_driver = MagicMock()
            retriever.driver = mock_driver
            retriever.database = None
            retriever.node_alias = "doc"

            session = MagicMock()
            result = MagicMock()
            result.__iter__ = Mock(return_value=iter([Mock(data=Mock(return_value={"id": "doc1"}))]))
            session.run.return_value = result
            mock_driver.session.return_value.__enter__ = Mock(return_value=session)
            mock_driver.session.return_value.__exit__ = Mock(return_value=False)

            results = retriever._execute_graph_query(
                query="MATCH (doc:Document) RETURN doc LIMIT 5",
                permission_filter=("", {}),
                limit=10
            )

            # Should not add another LIMIT
            call_args = session.run.call_args[0][0]
            assert call_args.count("LIMIT") == 1


class TestNeo4jVectorSearch:
    """Tests for Neo4j _execute_vector_search method."""

    def test_vector_search_with_permission_filter(self):
        """Test vector search with permission filter."""
        from ragguard.retrievers.neo4j import Neo4jSecureRetriever

        with patch.object(Neo4jSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = Neo4jSecureRetriever.__new__(Neo4jSecureRetriever)
            mock_driver = MagicMock()
            retriever.driver = mock_driver
            retriever.database = None
            retriever.node_alias = "doc"
            retriever.node_label = "Document"

            session = MagicMock()
            result = MagicMock()
            result.__iter__ = Mock(return_value=iter([
                Mock(data=Mock(return_value={"doc": {"id": "doc1"}, "score": 0.95}))
            ]))
            session.run.return_value = result
            mock_driver.session.return_value.__enter__ = Mock(return_value=session)
            mock_driver.session.return_value.__exit__ = Mock(return_value=False)

            results = retriever._execute_vector_search(
                vector=[0.1, 0.2, 0.3],
                permission_filter=("doc.department = $dept", {"dept": "eng"}),
                limit=10
            )

            call_args = session.run.call_args[0][0]
            assert "db.index.vector.queryNodes" in call_args
            assert "WHERE" in call_args

    def test_vector_search_without_permission_filter(self):
        """Test vector search without permission filter."""
        from ragguard.retrievers.neo4j import Neo4jSecureRetriever

        with patch.object(Neo4jSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = Neo4jSecureRetriever.__new__(Neo4jSecureRetriever)
            mock_driver = MagicMock()
            retriever.driver = mock_driver
            retriever.database = None
            retriever.node_alias = "doc"
            retriever.node_label = "Document"

            session = MagicMock()
            result = MagicMock()
            result.__iter__ = Mock(return_value=iter([
                Mock(data=Mock(return_value={"doc": {"id": "doc1"}, "score": 0.95}))
            ]))
            session.run.return_value = result
            mock_driver.session.return_value.__enter__ = Mock(return_value=session)
            mock_driver.session.return_value.__exit__ = Mock(return_value=False)

            results = retriever._execute_vector_search(
                vector=[0.1, 0.2, 0.3],
                permission_filter=("", {}),
                limit=10
            )

            call_args = session.run.call_args[0][0]
            assert "db.index.vector.queryNodes" in call_args
            # No WHERE clause without permission filter
            assert call_args.count("WHERE") == 0 or "WHERE" not in call_args


class TestNeo4jTraversal:
    """Tests for Neo4j _execute_traversal method."""

    def test_traversal_incoming(self):
        """Test incoming traversal."""
        from ragguard.retrievers.neo4j import Neo4jSecureRetriever

        with patch.object(Neo4jSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = Neo4jSecureRetriever.__new__(Neo4jSecureRetriever)
            mock_driver = MagicMock()
            retriever.driver = mock_driver
            retriever.database = None
            retriever.node_alias = "doc"
            retriever.node_label = "Document"

            retriever._build_permission_clause = Mock(return_value=("", {}))

            session = MagicMock()
            result = MagicMock()
            result.__iter__ = Mock(return_value=iter([Mock(data=Mock(return_value={"id": "doc1"}))]))
            session.run.return_value = result
            mock_driver.session.return_value.__enter__ = Mock(return_value=session)
            mock_driver.session.return_value.__exit__ = Mock(return_value=False)

            results = retriever._execute_traversal(
                start_node_id="doc1",
                relationship_type="RELATES_TO",
                user={"id": "alice"},
                direction="incoming",
                depth=2,
                limit=10
            )

            call_args = session.run.call_args[0][0]
            assert "<-[:RELATES_TO*1..2]-" in call_args

    def test_traversal_both_directions(self):
        """Test bidirectional traversal."""
        from ragguard.retrievers.neo4j import Neo4jSecureRetriever

        with patch.object(Neo4jSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = Neo4jSecureRetriever.__new__(Neo4jSecureRetriever)
            mock_driver = MagicMock()
            retriever.driver = mock_driver
            retriever.database = None
            retriever.node_alias = "doc"
            retriever.node_label = "Document"

            retriever._build_permission_clause = Mock(return_value=("", {}))

            session = MagicMock()
            result = MagicMock()
            result.__iter__ = Mock(return_value=iter([Mock(data=Mock(return_value={"id": "doc1"}))]))
            session.run.return_value = result
            mock_driver.session.return_value.__enter__ = Mock(return_value=session)
            mock_driver.session.return_value.__exit__ = Mock(return_value=False)

            results = retriever._execute_traversal(
                start_node_id="doc1",
                relationship_type="RELATES_TO",
                user={"id": "alice"},
                direction="both",
                depth=1,
                limit=10
            )

            call_args = session.run.call_args[0][0]
            assert "-[:RELATES_TO*1..1]-" in call_args


class TestNeo4jHealthCheckAndExit:
    """Tests for Neo4j health check and context manager."""

    def test_health_check(self):
        """Test health check."""
        from ragguard.retrievers.neo4j import Neo4jSecureRetriever

        with patch.object(Neo4jSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = Neo4jSecureRetriever.__new__(Neo4jSecureRetriever)
            mock_driver = MagicMock()
            retriever.driver = mock_driver
            retriever.database = "test_db"
            retriever.node_label = "Document"

            session = MagicMock()
            result = MagicMock()
            result.__iter__ = Mock(return_value=iter([
                {"name": "Neo4j", "versions": ["5.15.0"]}
            ]))
            session.run.return_value = result
            mock_driver.session.return_value.__enter__ = Mock(return_value=session)
            mock_driver.session.return_value.__exit__ = Mock(return_value=False)

            health = retriever._check_backend_health()

            assert health["connection_alive"] is True
            assert health["database"] == "test_db"

    def test_get_session_with_database(self):
        """Test _get_session with specific database."""
        from ragguard.retrievers.neo4j import Neo4jSecureRetriever

        with patch.object(Neo4jSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = Neo4jSecureRetriever.__new__(Neo4jSecureRetriever)
            mock_driver = MagicMock()
            retriever.driver = mock_driver
            retriever.database = "my_database"

            retriever._get_session()

            mock_driver.session.assert_called_once_with(database="my_database")

    def test_get_session_without_database(self):
        """Test _get_session without specific database."""
        from ragguard.retrievers.neo4j import Neo4jSecureRetriever

        with patch.object(Neo4jSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = Neo4jSecureRetriever.__new__(Neo4jSecureRetriever)
            mock_driver = MagicMock()
            retriever.driver = mock_driver
            retriever.database = None

            retriever._get_session()

            mock_driver.session.assert_called_once_with()

    def test_exit_closes_driver(self):
        """Test __exit__ closes the driver."""
        from ragguard.retrievers.neo4j import Neo4jSecureRetriever

        with patch.object(Neo4jSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = Neo4jSecureRetriever.__new__(Neo4jSecureRetriever)
            mock_driver = MagicMock()
            retriever.driver = mock_driver

            with patch.object(Neo4jSecureRetriever.__bases__[0], '__exit__', return_value=None):
                retriever.__exit__(None, None, None)

            mock_driver.close.assert_called_once()

    def test_exit_handles_close_error(self):
        """Test __exit__ handles errors when closing driver."""
        from ragguard.retrievers.neo4j import Neo4jSecureRetriever

        with patch.object(Neo4jSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = Neo4jSecureRetriever.__new__(Neo4jSecureRetriever)
            mock_driver = MagicMock()
            mock_driver.close.side_effect = Exception("Close error")
            retriever.driver = mock_driver

            with patch.object(Neo4jSecureRetriever.__bases__[0], '__exit__', return_value=None):
                # Should not raise
                retriever.__exit__(None, None, None)


class TestNeptuneMatchesSingleFilter:
    """Extended tests for Neptune _matches_single_filter method."""

    def test_missing_property_with_exists_false(self):
        """Test missing property with exists=False predicate."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)

            vertex = {"name": "doc1"}

            # Property doesn't exist, exists predicate is False - should match
            filter_spec = {"type": "has", "property": "deleted", "predicate": "exists", "value": False}
            assert retriever._matches_single_filter(vertex, filter_spec)

    def test_not_containing_predicate(self):
        """Test notContaining predicate."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)

            vertex = {"tags": ["important", "review"]}

            # Value not in list - should match
            filter_spec = {"type": "has", "property": "tags", "predicate": "notContaining", "value": "archived"}
            assert retriever._matches_single_filter(vertex, filter_spec)

            # Value in list - should not match
            filter_spec2 = {"type": "has", "property": "tags", "predicate": "notContaining", "value": "important"}
            assert not retriever._matches_single_filter(vertex, filter_spec2)

    def test_not_containing_non_list(self):
        """Test notContaining predicate on non-list value."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)

            vertex = {"category": "engineering"}

            # Non-list value defaults to True
            filter_spec = {"type": "has", "property": "category", "predicate": "notContaining", "value": "archived"}
            assert retriever._matches_single_filter(vertex, filter_spec)

    def test_containing_non_list(self):
        """Test containing predicate on non-list value."""
        from ragguard.retrievers.neptune import NeptuneSecureRetriever

        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)

            vertex = {"category": "engineering"}

            # Non-list value returns False
            filter_spec = {"type": "has", "property": "category", "predicate": "containing", "value": "eng"}
            assert not retriever._matches_single_filter(vertex, filter_spec)
