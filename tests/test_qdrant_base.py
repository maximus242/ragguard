"""
Qdrant retriever tests using the shared base test class.

This ensures Qdrant has consistent test coverage with other backends.
"""

from typing import Any, Dict, List

import pytest

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from base_retriever_tests import VectorDBRetrieverTestBase, VectorDBSecurityTestBase

from ragguard import Policy, QdrantSecureRetriever
from ragguard.audit import NullAuditLogger


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
class TestQdrantRetrieverBase(VectorDBRetrieverTestBase):
    """Qdrant implementation of the base retriever tests."""

    backend_name = "qdrant"
    retriever_class = QdrantSecureRetriever
    vector_size = 128

    def create_client(self) -> Any:
        """Create an in-memory Qdrant client."""
        return QdrantClient(":memory:")

    def create_collection(self, client: Any, name: str, vector_size: int) -> str:
        """Create a Qdrant collection."""
        # Check if collection exists, delete if so
        collections = client.get_collections().collections
        if any(c.name == name for c in collections):
            client.delete_collection(name)

        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        return name

    def insert_documents(self, client: Any, collection: str, documents: List[Dict]) -> None:
        """Insert documents into Qdrant collection."""
        points = []
        for i, doc in enumerate(documents):
            point = PointStruct(
                id=i + 1,
                vector=self.generate_embedding(seed=i),
                payload={
                    "id": doc["id"],
                    "text": doc["text"],
                    "visibility": doc.get("visibility"),
                    "department": doc.get("department"),
                    "confidential": doc.get("confidential", False),
                },
            )
            points.append(point)

        client.upsert(collection_name=collection, points=points)

    def create_retriever(self, client: Any, collection: str, policy: Policy, **kwargs) -> Any:
        """Create a QdrantSecureRetriever."""
        return QdrantSecureRetriever(
            client=client,
            collection=collection,
            policy=policy,
            audit_logger=NullAuditLogger(),
            **kwargs
        )

    def _get_metadata(self, result: Any) -> Dict:
        """Extract metadata from Qdrant ScoredPoint."""
        return result.payload


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
class TestQdrantRetrieverSecurity(VectorDBSecurityTestBase, TestQdrantRetrieverBase):
    """Qdrant security-focused tests."""
    pass
