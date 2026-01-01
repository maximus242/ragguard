"""
Tests for Qdrant secure retriever.

These tests require qdrant-client to be installed.
"""

import pytest

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from ragguard import Policy, QdrantSecureRetriever
from ragguard.audit import NullAuditLogger


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
class TestQdrantSecureRetriever:
    """Tests for Qdrant secure retriever."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create in-memory Qdrant client
        self.client = QdrantClient(":memory:")

        # Create test collection
        self.client.create_collection(
            collection_name="test_docs",
            vectors_config=VectorParams(size=128, distance=Distance.COSINE),
        )

        # Insert test documents
        import random
        random.seed(42)

        def fake_embed(seed):
            random.seed(seed)
            return [random.random() for _ in range(128)]

        test_docs = [
            {
                "id": 1,
                "text": "Public announcement",
                "visibility": "public",
            },
            {
                "id": 2,
                "text": "Engineering doc",
                "department": "engineering",
                "confidential": False,
            },
            {
                "id": 3,
                "text": "Finance report",
                "department": "finance",
                "confidential": True,
            },
        ]

        self.client.upsert(
            collection_name="test_docs",
            points=[
                PointStruct(
                    id=doc["id"],
                    vector=fake_embed(doc["id"]),
                    payload=doc,
                )
                for doc in test_docs
            ],
        )

    def test_search_public_documents(self):
        """Test that everyone can see public documents."""
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

        retriever = QdrantSecureRetriever(
            client=self.client,
            collection="test_docs",
            policy=policy,
            audit_logger=NullAuditLogger(),
        )

        user = {"id": "test@example.com"}

        import random
        random.seed(1)
        query = [random.random() for _ in range(128)]

        results = retriever.search(query=query, user=user, limit=10)

        # Should only return public documents
        assert len(results) == 1
        assert results[0].payload["visibility"] == "public"

    def test_search_department_documents(self):
        """Test department-scoped access."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "dept-docs",
                    "match": {"confidential": False},
                    "allow": {
                        "conditions": ["user.department == document.department"]
                    },
                }
            ],
            "default": "deny",
        })

        retriever = QdrantSecureRetriever(
            client=self.client,
            collection="test_docs",
            policy=policy,
            audit_logger=NullAuditLogger(),
        )

        engineer_user = {
            "id": "eng@example.com",
            "department": "engineering"
        }

        import random
        random.seed(2)
        query = [random.random() for _ in range(128)]

        results = retriever.search(query=query, user=engineer_user, limit=10)

        # Should only return engineering non-confidential docs
        for r in results:
            assert r.payload["department"] == "engineering"
            assert r.payload["confidential"] is False

    def test_search_admin_access(self):
        """Test that admins can see everything."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "admin",
                    "allow": {"roles": ["admin"]},
                }
            ],
            "default": "deny",
        })

        retriever = QdrantSecureRetriever(
            client=self.client,
            collection="test_docs",
            policy=policy,
            audit_logger=NullAuditLogger(),
        )

        admin_user = {
            "id": "admin@example.com",
            "roles": ["admin"]
        }

        import random
        random.seed(3)
        query = [random.random() for _ in range(128)]

        results = retriever.search(query=query, user=admin_user, limit=10)

        # Admin should see all documents
        assert len(results) == 3

    def test_search_no_access(self):
        """Test that users with no matching rules see nothing."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "admin-only",
                    "allow": {"roles": ["admin"]},
                }
            ],
            "default": "deny",
        })

        retriever = QdrantSecureRetriever(
            client=self.client,
            collection="test_docs",
            policy=policy,
            audit_logger=NullAuditLogger(),
        )

        regular_user = {
            "id": "user@example.com",
            "roles": ["user"]
        }

        import random
        random.seed(4)
        query = [random.random() for _ in range(128)]

        results = retriever.search(query=query, user=regular_user, limit=10)

        # Should return no results
        assert len(results) == 0

    def test_backend_name(self):
        """Test that backend name is correct."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "test", "allow": {"everyone": True}}],
        })

        retriever = QdrantSecureRetriever(
            client=self.client,
            collection="test_docs",
            policy=policy,
        )

        assert retriever.backend_name == "qdrant"
