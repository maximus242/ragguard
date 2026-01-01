"""
Base test class for vector database retrievers.

This module provides a shared test base that ensures consistent test coverage
across all vector database backends. Each backend should inherit from these
base classes and implement the abstract setup methods.

Usage:
    class TestMyDBRetriever(VectorDBRetrieverTestBase):
        backend_name = "mydb"
        retriever_class = MyDBSecureRetriever

        def create_client(self):
            return MyDBClient()

        def create_collection(self, client, name, vector_size):
            # Create collection in your DB
            pass

        def insert_documents(self, client, collection, documents):
            # Insert test documents
            pass
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from unittest.mock import Mock

import pytest

from ragguard import Policy
from ragguard.audit import NullAuditLogger


class VectorDBRetrieverTestBase(ABC):
    """
    Base test class for vector database secure retrievers.

    Subclasses must implement:
    - backend_name: str
    - retriever_class: Type
    - create_client() -> client
    - create_collection(client, name, vector_size) -> collection
    - insert_documents(client, collection, documents) -> None
    - create_retriever(client, collection, policy, **kwargs) -> retriever
    """

    # Subclasses must define these
    backend_name: str = None
    retriever_class: Type = None
    vector_size: int = 128

    # Test documents - same across all backends
    TEST_DOCUMENTS = [
        {
            "id": "doc1",
            "text": "Public announcement for everyone",
            "visibility": "public",
            "department": None,
            "confidential": False,
        },
        {
            "id": "doc2",
            "text": "Engineering technical documentation",
            "visibility": "internal",
            "department": "engineering",
            "confidential": False,
        },
        {
            "id": "doc3",
            "text": "Confidential finance report Q4",
            "visibility": "internal",
            "department": "finance",
            "confidential": True,
        },
        {
            "id": "doc4",
            "text": "HR policy document",
            "visibility": "internal",
            "department": "hr",
            "confidential": False,
        },
    ]

    @abstractmethod
    def create_client(self) -> Any:
        """Create and return a database client."""
        pass

    @abstractmethod
    def create_collection(self, client: Any, name: str, vector_size: int) -> Any:
        """Create a collection/index in the database."""
        pass

    @abstractmethod
    def insert_documents(self, client: Any, collection: Any, documents: List[Dict]) -> None:
        """Insert test documents into the collection."""
        pass

    @abstractmethod
    def create_retriever(self, client: Any, collection: Any, policy: Policy, **kwargs) -> Any:
        """Create a retriever instance."""
        pass

    def generate_embedding(self, seed: int) -> List[float]:
        """Generate a deterministic fake embedding."""
        import random
        random.seed(seed)
        return [random.random() for _ in range(self.vector_size)]

    def generate_query_embedding(self) -> List[float]:
        """Generate a query embedding."""
        return self.generate_embedding(seed=999)

    @pytest.fixture
    def client(self):
        """Create a database client."""
        return self.create_client()

    @pytest.fixture
    def collection(self, client):
        """Create a test collection with documents."""
        coll = self.create_collection(client, "test_docs", self.vector_size)
        self.insert_documents(client, coll, self.TEST_DOCUMENTS)
        return coll

    # ============================================================
    # CORE TESTS - Every backend must pass these
    # ============================================================

    def test_backend_name(self, client, collection):
        """Test that backend_name property returns correct value."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "all", "allow": {"everyone": True}}],
        })
        retriever = self.create_retriever(client, collection, policy)
        assert retriever.backend_name == self.backend_name

    def test_search_public_documents(self, client, collection):
        """Test that everyone can access public documents."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "public",
                    "match": {"visibility": "public"},
                    "allow": {"everyone": True},
                }
            ],
            "default": "deny",
        })

        retriever = self.create_retriever(client, collection, policy)
        results = retriever.search(
            query=self.generate_query_embedding(),
            user={"id": "anonymous"},
            limit=10
        )

        # Should only return public documents
        assert len(results) >= 1
        for r in results:
            metadata = self._get_metadata(r)
            assert metadata.get("visibility") == "public"

    def test_search_department_access(self, client, collection):
        """Test department-scoped access control."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "dept-access",
                    "allow": {
                        "conditions": ["user.department == document.department"]
                    },
                }
            ],
            "default": "deny",
        })

        retriever = self.create_retriever(client, collection, policy)
        results = retriever.search(
            query=self.generate_query_embedding(),
            user={"id": "eng@company.com", "department": "engineering"},
            limit=10
        )

        # Should only return engineering documents
        for r in results:
            metadata = self._get_metadata(r)
            assert metadata.get("department") == "engineering"

    def test_search_role_based_access(self, client, collection):
        """Test role-based access control."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "admin-all",
                    "allow": {"roles": ["admin"]},
                }
            ],
            "default": "deny",
        })

        retriever = self.create_retriever(client, collection, policy)

        # Admin should see all documents
        admin_results = retriever.search(
            query=self.generate_query_embedding(),
            user={"id": "admin@company.com", "roles": ["admin"]},
            limit=10
        )
        assert len(admin_results) == len(self.TEST_DOCUMENTS)

        # Regular user should see nothing
        user_results = retriever.search(
            query=self.generate_query_embedding(),
            user={"id": "user@company.com", "roles": ["user"]},
            limit=10
        )
        assert len(user_results) == 0

    def test_search_deny_all_default(self, client, collection):
        """Test that deny default blocks access when no rules match."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "impossible",
                    "match": {"visibility": "nonexistent"},
                    "allow": {"everyone": True},
                }
            ],
            "default": "deny",
        })

        retriever = self.create_retriever(client, collection, policy)
        results = retriever.search(
            query=self.generate_query_embedding(),
            user={"id": "user@company.com"},
            limit=10
        )

        # Should return no results
        assert len(results) == 0

    def test_search_with_limit(self, client, collection):
        """Test that limit parameter is respected."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "all", "allow": {"everyone": True}}],
        })

        retriever = self.create_retriever(client, collection, policy)

        # Request only 2 results
        results = retriever.search(
            query=self.generate_query_embedding(),
            user={"id": "user@company.com"},
            limit=2
        )

        assert len(results) <= 2

    def test_search_empty_collection(self, client):
        """Test search on empty collection returns empty results."""
        empty_coll = self.create_collection(client, "empty_docs", self.vector_size)

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "all", "allow": {"everyone": True}}],
        })

        retriever = self.create_retriever(client, empty_coll, policy)
        results = retriever.search(
            query=self.generate_query_embedding(),
            user={"id": "user@company.com"},
            limit=10
        )

        assert len(results) == 0

    def test_policy_update(self, client, collection):
        """Test that policy can be updated."""
        # Start with restrictive policy
        policy1 = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "none", "allow": {"roles": ["nonexistent"]}}],
            "default": "deny",
        })

        retriever = self.create_retriever(client, collection, policy1)
        results1 = retriever.search(
            query=self.generate_query_embedding(),
            user={"id": "user@company.com"},
            limit=10
        )
        assert len(results1) == 0

        # Update to permissive policy
        policy2 = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "all", "allow": {"everyone": True}}],
        })
        retriever.policy = policy2

        results2 = retriever.search(
            query=self.generate_query_embedding(),
            user={"id": "user@company.com"},
            limit=10
        )
        assert len(results2) > 0

    def test_health_check(self, client, collection):
        """Test health check returns healthy status."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "all", "allow": {"everyone": True}}],
        })

        retriever = self.create_retriever(client, collection, policy)
        health = retriever.health_check()

        assert health["healthy"] is True
        assert health["backend"] == self.backend_name

    # ============================================================
    # HELPER METHODS
    # ============================================================

    def _get_metadata(self, result: Any) -> Dict:
        """Extract metadata from a search result (backend-specific)."""
        # Default implementation - subclasses may override
        if hasattr(result, 'payload'):
            return result.payload
        if hasattr(result, 'metadata'):
            return result.metadata
        if isinstance(result, dict):
            return result.get('metadata', result.get('payload', result))
        return {}


class MockedVectorDBTestBase(VectorDBRetrieverTestBase):
    """
    Base class for backends that use mocked clients.

    Use this for backends where setting up a real client is complex
    or requires external services.
    """

    def create_mock_client(self) -> Mock:
        """Create a mock client."""
        return Mock()

    def configure_mock_search_response(self, mock_client: Mock, results: List[Dict]) -> None:
        """Configure mock to return specific search results."""
        pass  # Subclasses implement based on their API


# ============================================================
# SECURITY-FOCUSED TESTS
# ============================================================

class VectorDBSecurityTestBase(VectorDBRetrieverTestBase):
    """
    Extended tests focusing on security scenarios.

    These tests verify that the retriever properly enforces
    access control in edge cases and attack scenarios.
    """

    def test_no_data_leakage_in_results(self, client, collection):
        """Test that unauthorized documents are never returned."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "dept-only",
                    "allow": {
                        "conditions": ["user.department == document.department"]
                    },
                }
            ],
            "default": "deny",
        })

        retriever = self.create_retriever(client, collection, policy)

        # Finance user should never see engineering docs
        results = retriever.search(
            query=self.generate_query_embedding(),
            user={"id": "finance@company.com", "department": "finance"},
            limit=100  # Request many to ensure no leakage
        )

        for r in results:
            metadata = self._get_metadata(r)
            dept = metadata.get("department")
            assert dept is None or dept == "finance", \
                f"Data leakage: finance user saw {dept} document"

    def test_confidential_documents_protected(self, client, collection):
        """Test that confidential documents require explicit access."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "non-confidential",
                    "match": {"confidential": False},
                    "allow": {"everyone": True},
                }
            ],
            "default": "deny",
        })

        retriever = self.create_retriever(client, collection, policy)
        results = retriever.search(
            query=self.generate_query_embedding(),
            user={"id": "user@company.com"},
            limit=100
        )

        for r in results:
            metadata = self._get_metadata(r)
            assert metadata.get("confidential") is not True, \
                "Confidential document was returned without authorization"

    def test_user_context_required(self, client, collection):
        """Test that search fails without user context."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "all", "allow": {"everyone": True}}],
        })

        retriever = self.create_retriever(client, collection, policy)

        # Search without user should raise error
        with pytest.raises(Exception):  # Could be ValueError, RetrieverError, etc.
            retriever.search(
                query=self.generate_query_embedding(),
                user=None,
                limit=10
            )
