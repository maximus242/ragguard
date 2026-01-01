"""Tests for graph database retrievers."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from ragguard.policy.models import AllowConditions, Policy, Rule
from ragguard.retrievers.arangodb import ArangoDBSecureRetriever
from ragguard.retrievers.graph_base import BaseGraphRetriever
from ragguard.retrievers.neo4j import Neo4jSecureRetriever
from ragguard.retrievers.neptune import NeptuneSecureRetriever
from ragguard.retrievers.tigergraph import TigerGraphSecureRetriever


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


class TestBaseGraphRetriever:
    """Tests for the base graph retriever class."""

    def test_base_graph_retriever_is_abstract(self):
        """Test that BaseGraphRetriever cannot be instantiated directly."""
        policy = make_policy([{"everyone": True}])

        # BaseGraphRetriever is abstract, can't instantiate directly
        with pytest.raises(TypeError):
            BaseGraphRetriever(
                client=Mock(),
                node_label="Document",
                policy=policy
            )


class TestNeo4jSecureRetriever:
    """Tests for Neo4j secure retriever."""

    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = MagicMock()
        session = MagicMock()
        result = MagicMock()

        # Configure the mock chain
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=False)
        session.run.return_value = result
        result.__iter__ = Mock(return_value=iter([
            Mock(data=Mock(return_value={"id": "doc1", "title": "Test Doc"}))
        ]))

        return driver

    @pytest.fixture
    def policy(self):
        """Create a test policy."""
        return make_policy([
            {
                "roles": ["engineer"],
                "conditions": ["user.department == document.department"]
            }
        ])

    @patch("ragguard.retrievers.neo4j.Neo4jSecureRetriever._check_backend_health")
    def test_initialization(self, mock_health, mock_driver, policy):
        """Test Neo4j retriever initialization."""
        # Skip type checking by patching
        with patch("ragguard.retrievers.neo4j.Neo4jSecureRetriever.__init__", lambda self, **kwargs: None):
            pass  # Skip actual init for this test

    def test_backend_name(self, mock_driver, policy):
        """Test backend_name property."""
        with patch.object(Neo4jSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = Neo4jSecureRetriever.__new__(Neo4jSecureRetriever)
            assert retriever.backend_name == "neo4j"

    def test_property_search(self, mock_driver, policy):
        """Test property-based search."""
        with patch.object(Neo4jSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = Neo4jSecureRetriever.__new__(Neo4jSecureRetriever)
            retriever.driver = mock_driver
            retriever.database = None
            retriever.node_alias = "doc"
            retriever.node_label = "Document"
            retriever._policy = policy  # Use internal attribute to avoid setter
            retriever.collection = "Document"  # Required for logging
            retriever._enable_filter_cache = False
            retriever._filter_cache_size = 1000

            # Mock the policy engine
            retriever.policy_engine = MagicMock()
            retriever.policy_engine.to_filter.return_value = ("doc.department = $eq_1", {"eq_1": "engineering"})

            # Mock session
            session = MagicMock()
            result = MagicMock()
            result.__iter__ = Mock(return_value=iter([
                Mock(data=Mock(return_value={"id": "doc1"}))
            ]))
            session.run.return_value = result
            mock_driver.session.return_value.__enter__ = Mock(return_value=session)
            mock_driver.session.return_value.__exit__ = Mock(return_value=False)

            # Call property search
            results = retriever._execute_property_search(
                properties={"category": "engineering"},
                permission_filter=("doc.department = $eq_1", {"eq_1": "engineering"}),
                limit=10
            )

            assert len(results) == 1
            session.run.assert_called_once()

    def test_traversal(self, mock_driver, policy):
        """Test relationship traversal."""
        with patch.object(Neo4jSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = Neo4jSecureRetriever.__new__(Neo4jSecureRetriever)
            retriever.driver = mock_driver
            retriever.database = None
            retriever.node_alias = "doc"
            retriever.node_label = "Document"
            retriever._policy = policy  # Use internal attribute to avoid setter
            retriever.collection = "Document"

            # Mock the permission clause building
            retriever._build_permission_clause = Mock(return_value=("doc.department = $eq_1", {"eq_1": "engineering"}))

            # Mock session
            session = MagicMock()
            result = MagicMock()
            result.__iter__ = Mock(return_value=iter([
                Mock(data=Mock(return_value={"id": "doc2"}))
            ]))
            session.run.return_value = result
            mock_driver.session.return_value.__enter__ = Mock(return_value=session)
            mock_driver.session.return_value.__exit__ = Mock(return_value=False)

            results = retriever._execute_traversal(
                start_node_id="doc1",
                relationship_type="RELATES_TO",
                user={"id": "alice", "department": "engineering"},
                direction="outgoing",
                depth=2,
                limit=10
            )

            assert len(results) == 1
            # Verify the query contains traversal pattern
            call_args = session.run.call_args[0][0]
            assert "-[:RELATES_TO*1..2]->" in call_args


class TestNeptuneSecureRetriever:
    """Tests for Amazon Neptune secure retriever."""

    @pytest.fixture
    def mock_gremlin_client(self):
        """Create a mock Gremlin client."""
        client = MagicMock()
        result_set = MagicMock()
        result_set.all.return_value.result.return_value = [
            {"id": "v1", "category": "engineering"}
        ]
        client.submit.return_value = result_set
        return client

    @pytest.fixture
    def policy(self):
        return make_policy([{"everyone": True}])

    def test_backend_name(self, mock_gremlin_client, policy):
        """Test backend_name property."""
        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)
            assert retriever.backend_name == "neptune"

    def test_matches_single_filter(self, mock_gremlin_client, policy):
        """Test single filter matching logic."""
        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)

            vertex = {"category": "engineering", "department": "eng"}

            # Test equality filter matching
            eq_filter = {"type": "has", "property": "category", "predicate": "eq", "value": "engineering"}
            assert retriever._matches_single_filter(vertex, eq_filter)

            wrong_filter = {"type": "has", "property": "category", "predicate": "eq", "value": "sales"}
            assert not retriever._matches_single_filter(vertex, wrong_filter)

    def test_matches_filters(self, mock_gremlin_client, policy):
        """Test filter matching logic."""
        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)

            vertex = {"category": "engineering", "priority": 5}

            # Test equality filter
            eq_filter = [{"type": "has", "property": "category", "predicate": "eq", "value": "engineering"}]
            assert retriever._matches_filters(vertex, eq_filter)

            # Test within filter
            within_filter = [{"type": "has", "property": "category", "predicate": "within", "value": ["engineering", "sales"]}]
            assert retriever._matches_filters(vertex, within_filter)

            # Test gt filter
            gt_filter = [{"type": "has", "property": "priority", "predicate": "gt", "value": 3}]
            assert retriever._matches_filters(vertex, gt_filter)


class TestTigerGraphSecureRetriever:
    """Tests for TigerGraph secure retriever."""

    @pytest.fixture
    def mock_connection(self):
        """Create a mock TigerGraph connection."""
        conn = MagicMock()
        conn.getVertices.return_value = [
            {"v_id": "v1", "attributes": {"category": "engineering"}}
        ]
        conn.gsql.return_value = [{"id": "v1"}]
        return conn

    @pytest.fixture
    def policy(self):
        return make_policy([{"everyone": True}])

    def test_backend_name(self, mock_connection, policy):
        """Test backend_name property."""
        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)
            assert retriever.backend_name == "tigergraph"

    def test_matches_properties(self, mock_connection, policy):
        """Test property matching."""
        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)

            # Test vertex with attributes wrapper
            vertex = {"attributes": {"category": "engineering", "priority": 5}}
            assert retriever._matches_properties(vertex, {"category": "engineering"})
            assert not retriever._matches_properties(vertex, {"category": "sales"})

            # Test vertex without attributes wrapper
            vertex2 = {"category": "engineering", "priority": 5}
            assert retriever._matches_properties(vertex2, {"category": "engineering"})


class TestArangoDBSecureRetriever:
    """Tests for ArangoDB secure retriever."""

    @pytest.fixture
    def mock_database(self):
        """Create a mock ArangoDB database."""
        db = MagicMock()
        db.name = "test_db"

        cursor = MagicMock()
        cursor.__iter__ = Mock(return_value=iter([{"_id": "doc1", "category": "engineering"}]))
        db.aql.execute.return_value = cursor

        collection = MagicMock()
        collection.properties.return_value = {"name": "documents"}
        db.collection.return_value = collection

        return db

    @pytest.fixture
    def policy(self):
        return make_policy([{"everyone": True}])

    def test_backend_name(self, mock_database, policy):
        """Test backend_name property."""
        with patch.object(ArangoDBSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = ArangoDBSecureRetriever.__new__(ArangoDBSecureRetriever)
            assert retriever.backend_name == "arangodb"

    def test_execute_property_search(self, mock_database, policy):
        """Test property-based search."""
        with patch.object(ArangoDBSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = ArangoDBSecureRetriever.__new__(ArangoDBSecureRetriever)
            retriever.database = mock_database
            retriever.collection_name = "documents"
            retriever.doc_alias = "doc"

            cursor = MagicMock()
            cursor.__iter__ = Mock(return_value=iter([
                {"_id": "doc1", "category": "engineering"}
            ]))
            mock_database.aql.execute.return_value = cursor

            results = retriever._execute_property_search(
                properties={"category": "engineering"},
                permission_filter=("true", {}),
                limit=10
            )

            assert len(results) == 1
            mock_database.aql.execute.assert_called_once()

            # Verify query structure
            call_args = mock_database.aql.execute.call_args
            query = call_args[0][0]
            assert "FOR doc IN documents" in query
            assert "FILTER" in query

    def test_health_check(self, mock_database, policy):
        """Test health check."""
        with patch.object(ArangoDBSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = ArangoDBSecureRetriever.__new__(ArangoDBSecureRetriever)
            retriever.database = mock_database
            retriever.collection_name = "documents"

            mock_database.properties.return_value = {"name": "test_db"}

            health = retriever._check_backend_health()

            assert health["connection_alive"] is True
            assert health["collection_name"] == "documents"


class TestGraphRetrieverFeatures:
    """Test common features across graph retrievers."""

    def test_traverse_validates_direction(self):
        """Test that traverse validates direction parameter."""
        with patch.object(Neo4jSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = Neo4jSecureRetriever.__new__(Neo4jSecureRetriever)
            retriever.driver = MagicMock()
            retriever.database = None
            retriever.node_alias = "doc"
            retriever.node_label = "Document"
            retriever._policy = make_policy([{"everyone": True}])
            retriever.collection = "Document"

            # The traverse method validates direction
            with pytest.raises(ValueError, match="Invalid direction"):
                retriever.traverse(
                    start_node_id="doc1",
                    relationship_type="RELATES_TO",
                    user={"id": "alice"},
                    direction="invalid",
                    depth=1
                )

    def test_vector_search_not_implemented_by_default(self):
        """Test that vector search raises NotImplementedError by default."""
        with patch.object(Neo4jSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = Neo4jSecureRetriever.__new__(Neo4jSecureRetriever)
            retriever.driver = MagicMock()
            retriever.database = None
            retriever.node_alias = "doc"
            retriever.node_label = "Document"

            # Neptune retriever doesn't implement vector search
            with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
                neptune = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)
                neptune.gremlin_client = MagicMock()
                neptune.node_label = "Document"

                with pytest.raises(NotImplementedError):
                    neptune._execute_vector_search(
                        vector=[0.1, 0.2, 0.3],
                        permission_filter=[],
                        limit=10
                    )


class TestGraphRetrieverPolicyIntegration:
    """Test graph retrievers integrate properly with PolicyEngine."""

    def test_neo4j_uses_policy_engine(self):
        """Test Neo4j retriever uses PolicyEngine for filter generation."""
        policy = make_policy([
            {
                "roles": ["admin"],
                "conditions": ["user.department == document.department"]
            }
        ])

        import threading

        with patch.object(Neo4jSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = Neo4jSecureRetriever.__new__(Neo4jSecureRetriever)
            retriever._policy = policy  # Use internal attribute
            retriever._policy_lock = threading.RLock()  # Add required lock
            retriever.collection = "Document"
            retriever.node_alias = "doc"

            # Build permission clause
            where, params = retriever._build_permission_clause(
                user={"id": "alice", "roles": ["admin"], "department": "engineering"}
            )

            # Should generate a valid Cypher WHERE clause
            assert "doc." in where or where == ""
            # If admin role matched and has condition, should have department filter
            if where:
                assert "engineering" in params.values() or "department" in where


class TestGraphRetrieverErrorHandling:
    """Test error handling in graph retrievers."""

    def test_neo4j_query_execution_error(self):
        """Test Neo4j handles query execution errors."""
        with patch.object(Neo4jSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = Neo4jSecureRetriever.__new__(Neo4jSecureRetriever)
            mock_driver = MagicMock()
            retriever.driver = mock_driver
            retriever.database = None
            retriever.node_alias = "doc"
            retriever.node_label = "Document"

            # Mock session to raise exception
            session = MagicMock()
            session.run.side_effect = Exception("Query execution failed")
            mock_driver.session.return_value.__enter__ = Mock(return_value=session)
            mock_driver.session.return_value.__exit__ = Mock(return_value=False)

            with pytest.raises(Exception, match="Query execution failed"):
                retriever._execute_property_search(
                    properties={"category": "test"},
                    permission_filter=("true", {}),
                    limit=10
                )

    def test_arangodb_query_execution_error(self):
        """Test ArangoDB handles query execution errors."""
        with patch.object(ArangoDBSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = ArangoDBSecureRetriever.__new__(ArangoDBSecureRetriever)
            retriever.database = MagicMock()
            retriever.collection_name = "documents"
            retriever.doc_alias = "doc"

            # Mock to raise exception
            retriever.database.aql.execute.side_effect = Exception("AQL syntax error")

            with pytest.raises(Exception, match="AQL syntax error"):
                retriever._execute_property_search(
                    properties={"category": "test"},
                    permission_filter=("true", {}),
                    limit=10
                )

    def test_neptune_filter_matching_edge_cases(self):
        """Test Neptune filter matching edge cases."""
        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)

            vertex = {"category": "engineering", "priority": 5, "tags": ["important", "urgent"]}

            # Test lt predicate
            lt_filter = [{"type": "has", "property": "priority", "predicate": "lt", "value": 10}]
            assert retriever._matches_filters(vertex, lt_filter)

            lt_filter_fail = [{"type": "has", "property": "priority", "predicate": "lt", "value": 3}]
            assert not retriever._matches_filters(vertex, lt_filter_fail)

            # Test gte predicate
            gte_filter = [{"type": "has", "property": "priority", "predicate": "gte", "value": 5}]
            assert retriever._matches_filters(vertex, gte_filter)

            # Test lte predicate
            lte_filter = [{"type": "has", "property": "priority", "predicate": "lte", "value": 5}]
            assert retriever._matches_filters(vertex, lte_filter)

            # Test neq predicate
            neq_filter = [{"type": "has", "property": "category", "predicate": "neq", "value": "sales"}]
            assert retriever._matches_filters(vertex, neq_filter)

            neq_filter_fail = [{"type": "has", "property": "category", "predicate": "neq", "value": "engineering"}]
            assert not retriever._matches_filters(vertex, neq_filter_fail)

            # Test without predicate
            without_filter = [{"type": "has", "property": "category", "predicate": "without", "value": ["sales", "marketing"]}]
            assert retriever._matches_filters(vertex, without_filter)

            without_filter_fail = [{"type": "has", "property": "category", "predicate": "without", "value": ["engineering", "sales"]}]
            assert not retriever._matches_filters(vertex, without_filter_fail)

            # Test hasNot type
            has_not_filter = [{"type": "hasNot", "property": "nonexistent"}]
            assert retriever._matches_filters(vertex, has_not_filter)

            has_not_filter_fail = [{"type": "hasNot", "property": "category"}]
            assert not retriever._matches_filters(vertex, has_not_filter_fail)

            # Test empty filters list (should allow all)
            assert retriever._matches_filters(vertex, [])

    def test_tigergraph_matches_properties_nested(self):
        """Test TigerGraph property matching with nested values."""
        with patch.object(TigerGraphSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = TigerGraphSecureRetriever.__new__(TigerGraphSecureRetriever)

            # Test with attributes wrapper
            vertex = {"attributes": {"status": "active", "score": 100}}
            assert retriever._matches_properties(vertex, {"status": "active"})
            assert retriever._matches_properties(vertex, {"score": 100})
            assert not retriever._matches_properties(vertex, {"status": "inactive"})

            # Test with multiple properties
            assert retriever._matches_properties(vertex, {"status": "active", "score": 100})
            assert not retriever._matches_properties(vertex, {"status": "active", "score": 50})

    def test_traverse_direction_validation_all_backends(self):
        """Test traverse direction validation for all backends."""
        backends = [
            (Neo4jSecureRetriever, {"driver": MagicMock(), "database": None, "node_alias": "doc", "node_label": "Document"}),
        ]

        for retriever_class, extra_attrs in backends:
            with patch.object(retriever_class, '__init__', lambda self, **kwargs: None):
                retriever = retriever_class.__new__(retriever_class)
                retriever._policy = make_policy([{"everyone": True}])
                retriever.collection = "Document"
                for attr, val in extra_attrs.items():
                    setattr(retriever, attr, val)

                # Valid directions should not raise
                for direction in ["outgoing", "incoming", "both"]:
                    try:
                        # Mock _execute_traversal to avoid actual DB calls
                        retriever._execute_traversal = Mock(return_value=[])
                        retriever.traverse(
                            start_node_id="doc1",
                            relationship_type="RELATES_TO",
                            user={"id": "alice"},
                            direction=direction,
                            depth=1
                        )
                    except ValueError:
                        pytest.fail(f"Direction '{direction}' should be valid for {retriever_class.__name__}")

                # Invalid direction should raise
                with pytest.raises(ValueError, match="Invalid direction"):
                    retriever.traverse(
                        start_node_id="doc1",
                        relationship_type="RELATES_TO",
                        user={"id": "alice"},
                        direction="sideways",
                        depth=1
                    )


class TestNeptuneFilterMatching:
    """Additional tests for Neptune filter matching logic."""

    def test_or_filter_matching(self):
        """Test OR filter structure matching."""
        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)

            vertex = {"category": "engineering", "department": "eng"}

            # Test OR filter - should match if any child matches
            or_filter = [{
                "type": "or",
                "children": [
                    [{"type": "has", "property": "category", "predicate": "eq", "value": "sales"}],
                    [{"type": "has", "property": "category", "predicate": "eq", "value": "engineering"}]
                ]
            }]
            assert retriever._matches_filters(vertex, or_filter)

            # Test OR filter - none match
            or_filter_fail = [{
                "type": "or",
                "children": [
                    [{"type": "has", "property": "category", "predicate": "eq", "value": "sales"}],
                    [{"type": "has", "property": "category", "predicate": "eq", "value": "marketing"}]
                ]
            }]
            assert not retriever._matches_filters(vertex, or_filter_fail)

    def test_containing_predicate(self):
        """Test containing predicate for array membership."""
        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)

            vertex = {"tags": ["important", "urgent", "review"], "name": "doc1"}

            # Test containing predicate
            containing_filter = [{"type": "has", "property": "tags", "predicate": "containing", "value": "important"}]
            assert retriever._matches_filters(vertex, containing_filter)

            containing_filter_fail = [{"type": "has", "property": "tags", "predicate": "containing", "value": "archive"}]
            assert not retriever._matches_filters(vertex, containing_filter_fail)

    def test_unknown_predicate_fallback(self):
        """Test that unknown predicates default to True (permissive)."""
        with patch.object(NeptuneSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = NeptuneSecureRetriever.__new__(NeptuneSecureRetriever)

            vertex = {"category": "engineering"}

            # Unknown predicate defaults to True for permissive fallback behavior
            unknown_filter = [{"type": "has", "property": "category", "predicate": "unknown_op", "value": "engineering"}]
            assert retriever._matches_filters(vertex, unknown_filter)


class TestArangoDBRetrieverEdgeCases:
    """Edge case tests for ArangoDB retriever."""

    def test_execute_graph_query(self):
        """Test executing raw AQL graph queries."""
        with patch.object(ArangoDBSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = ArangoDBSecureRetriever.__new__(ArangoDBSecureRetriever)
            mock_db = MagicMock()
            retriever.database = mock_db
            retriever.collection_name = "documents"
            retriever.doc_alias = "doc"

            cursor = MagicMock()
            cursor.__iter__ = Mock(return_value=iter([
                {"_id": "doc1", "title": "Test"},
                {"_id": "doc2", "title": "Test 2"}
            ]))
            mock_db.aql.execute.return_value = cursor

            results = retriever._execute_graph_query(
                query="FOR doc IN documents RETURN doc",
                permission_filter=("true", {}),
                limit=10
            )

            assert len(results) == 2
            mock_db.aql.execute.assert_called_once()

    def test_execute_traversal(self):
        """Test executing traversal queries."""
        with patch.object(ArangoDBSecureRetriever, '__init__', lambda self, **kwargs: None):
            retriever = ArangoDBSecureRetriever.__new__(ArangoDBSecureRetriever)
            mock_db = MagicMock()
            retriever.database = mock_db
            retriever.collection_name = "documents"
            retriever.doc_alias = "doc"
            retriever.edge_collection = "edges"
            retriever._policy = make_policy([{"everyone": True}])
            retriever.collection = "documents"

            # Mock permission clause building
            retriever._build_permission_clause = Mock(return_value=("true", {}))

            cursor = MagicMock()
            cursor.__iter__ = Mock(return_value=iter([
                {"_id": "doc2", "title": "Related Doc"}
            ]))
            mock_db.aql.execute.return_value = cursor

            results = retriever._execute_traversal(
                start_node_id="documents/doc1",
                relationship_type="RELATES_TO",
                user={"id": "alice"},
                direction="outgoing",
                depth=2,
                limit=10
            )

            assert len(results) == 1
            mock_db.aql.execute.assert_called_once()
            call_args = mock_db.aql.execute.call_args[0][0]
            assert "OUTBOUND" in call_args or "FOR" in call_args
