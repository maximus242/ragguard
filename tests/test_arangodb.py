"""
Comprehensive tests for ArangoDB retriever to maximize coverage.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestArangoDBSecureRetriever:
    """Tests for ArangoDBSecureRetriever class."""

    @pytest.fixture
    def mock_policy(self):
        from ragguard.policy.models import Policy
        return Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "dept_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.department == document.department"]
                    }
                }
            ],
            "default": "deny"
        })

    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.name = "testdb"
        db.properties.return_value = {"name": "testdb", "isSystem": False}
        collection_mock = MagicMock()
        collection_mock.properties.return_value = {"name": "documents", "count": 100}
        db.collection.return_value = collection_mock
        return db

    def test_creation(self, mock_db, mock_policy):
        """Test retriever creation."""
        from ragguard.retrievers.arangodb import ArangoDBSecureRetriever

        retriever = ArangoDBSecureRetriever(
            database=mock_db,
            collection_name="documents",
            policy=mock_policy
        )

        assert retriever.collection_name == "documents"
        assert retriever.doc_alias == "doc"
        assert retriever.backend_name == "arangodb"

    def test_creation_with_custom_options(self, mock_db, mock_policy):
        """Test creation with custom options."""
        from ragguard.retrievers.arangodb import ArangoDBSecureRetriever

        retriever = ArangoDBSecureRetriever(
            database=mock_db,
            collection_name="docs",
            policy=mock_policy,
            doc_alias="d",
            edge_collection="edges",
            enable_filter_cache=False,
            enable_retry=False
        )

        assert retriever.doc_alias == "d"
        assert retriever.edge_collection == "edges"

    def test_backend_name(self, mock_db, mock_policy):
        """Test backend_name property."""
        from ragguard.retrievers.arangodb import ArangoDBSecureRetriever

        retriever = ArangoDBSecureRetriever(
            database=mock_db,
            collection_name="documents",
            policy=mock_policy
        )

        assert retriever.backend_name == "arangodb"

    def test_traversal_requires_edge_collection(self, mock_db, mock_policy):
        """Test traversal fails without edge collection."""
        from ragguard.exceptions import RetrieverError
        from ragguard.retrievers.arangodb import ArangoDBSecureRetriever

        retriever = ArangoDBSecureRetriever(
            database=mock_db,
            collection_name="documents",
            policy=mock_policy
            # No edge_collection
        )

        with pytest.raises(RetrieverError, match="edge_collection"):
            retriever._execute_traversal(
                start_node_id="doc/1",
                relationship_type="RELATES_TO",
                user={"id": "alice"},
                direction="outgoing",
                depth=2,
                limit=10
            )


class TestArangoDBFilterBuilder:
    """Tests for ArangoDB filter builder."""

    def test_to_arangodb_filter_simple(self):
        """Test to_arangodb_filter with simple policy."""
        from ragguard.filters.backends.arangodb import to_arangodb_filter
        from ragguard.policy.models import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "dept_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.department == document.department"]
                    }
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"], "department": "engineering"}

        filter_clause, bind_vars = to_arangodb_filter(policy, user)

        assert filter_clause is not None
        assert isinstance(filter_clause, str)

    def test_to_arangodb_filter_with_match(self):
        """Test to_arangodb_filter with match conditions."""
        from ragguard.filters.backends.arangodb import to_arangodb_filter
        from ragguard.policy.models import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "public_rule",
                    "allow": {"roles": ["user"]},
                    "match": {"visibility": "public"}
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"]}

        filter_clause, bind_vars = to_arangodb_filter(policy, user)

        assert filter_clause is not None
        assert "visibility" in filter_clause

    def test_to_arangodb_filter_default_allow(self):
        """Test to_arangodb_filter with default allow."""
        from ragguard.filters.backends.arangodb import to_arangodb_filter
        from ragguard.policy.models import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "admin_rule",
                    "allow": {"roles": ["admin"]}
                }
            ],
            "default": "allow"
        })

        user = {"id": "alice", "roles": ["user"]}

        filter_clause, bind_vars = to_arangodb_filter(policy, user)

        # Default allow when no rules match
        assert filter_clause == "true"

    def test_to_arangodb_filter_multiple_rules(self):
        """Test to_arangodb_filter with multiple rules."""
        from ragguard.filters.backends.arangodb import to_arangodb_filter
        from ragguard.policy.models import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "public_rule",
                    "allow": {"roles": ["user"]},
                    "match": {"visibility": "public"}
                },
                {
                    "name": "dept_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.department == document.department"]
                    }
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"], "department": "eng"}

        filter_clause, bind_vars = to_arangodb_filter(policy, user)

        assert filter_clause is not None
        # Should combine with OR
        assert "||" in filter_clause or "OR" in filter_clause.upper()

    def test_to_arangodb_filter_with_list_match(self):
        """Test to_arangodb_filter with list match values."""
        from ragguard.filters.backends.arangodb import to_arangodb_filter
        from ragguard.policy.models import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "category_rule",
                    "allow": {"roles": ["user"]},
                    "match": {"category": ["public", "internal"]}
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"]}

        filter_clause, bind_vars = to_arangodb_filter(policy, user)

        assert filter_clause is not None

    def test_to_arangodb_filter_custom_doc_alias(self):
        """Test to_arangodb_filter with custom document alias."""
        from ragguard.filters.backends.arangodb import to_arangodb_filter
        from ragguard.policy.models import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "dept_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.department == document.department"]
                    }
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"], "department": "eng"}

        filter_clause, bind_vars = to_arangodb_filter(policy, user, doc_alias="d")

        assert filter_clause is not None
        assert "d." in filter_clause  # Custom alias used

    def test_to_arangodb_filter_no_matching_rules(self):
        """Test to_arangodb_filter when no rules match."""
        from ragguard.filters.backends.arangodb import to_arangodb_filter
        from ragguard.policy.models import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "admin_rule",
                    "allow": {"roles": ["admin"]},
                    "match": {"type": "secret"}
                }
            ],
            "default": "deny"
        })

        user = {"id": "alice", "roles": ["user"]}

        filter_clause, bind_vars = to_arangodb_filter(policy, user)

        # Deny all when no rules match
        assert filter_clause is not None
        assert "false" in filter_clause.lower()
