"""
Tests to improve coverage for ragguard/types.py to 95%+.

Focuses on:
- StandardSearchResult conversion methods (from_qdrant, from_chromadb, etc.)
- validate_vector_dimension function
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


class TestStandardSearchResultConversions:
    """Tests for StandardSearchResult conversion class methods."""

    def test_from_qdrant_basic(self):
        """Test converting Qdrant ScoredPoint to StandardSearchResult."""
        from ragguard.types import StandardSearchResult

        # Mock Qdrant ScoredPoint
        point = MagicMock()
        point.id = "doc123"
        point.score = 0.95
        point.payload = {"text": "Hello world", "category": "greeting"}
        point.vector = None

        result = StandardSearchResult.from_qdrant(point)

        assert result.id == "doc123"
        assert result.score == 0.95
        assert result.metadata == {"text": "Hello world", "category": "greeting"}
        assert result.text == "Hello world"

    def test_from_qdrant_with_vector(self):
        """Test from_qdrant when vector is available."""
        from ragguard.types import StandardSearchResult

        point = MagicMock()
        point.id = 456
        point.score = 0.88
        point.payload = {"text": "Test document"}
        point.vector = [0.1, 0.2, 0.3]

        result = StandardSearchResult.from_qdrant(point)

        assert result.id == "456"
        assert result.vector == [0.1, 0.2, 0.3]

    def test_from_qdrant_no_payload(self):
        """Test from_qdrant when payload is None."""
        from ragguard.types import StandardSearchResult

        point = MagicMock()
        point.id = "doc1"
        point.score = 0.5
        point.payload = None
        point.vector = None

        result = StandardSearchResult.from_qdrant(point)

        assert result.id == "doc1"
        assert result.metadata == {}
        assert result.text is None

    def test_from_chromadb_basic(self):
        """Test converting ChromaDB result to StandardSearchResult."""
        from ragguard.types import StandardSearchResult

        result_dict = {
            "id": "chroma_doc_1",
            "distance": 0.5,
            "metadata": {"author": "Alice", "year": 2024},
            "document": "This is a ChromaDB document"
        }

        result = StandardSearchResult.from_chromadb(result_dict)

        assert result.id == "chroma_doc_1"
        # Score should be 1/(1+distance)
        assert abs(result.score - (1.0 / 1.5)) < 0.001
        assert result.metadata == {"author": "Alice", "year": 2024}
        assert result.text == "This is a ChromaDB document"
        assert result.vector is None

    def test_from_chromadb_no_distance(self):
        """Test from_chromadb when distance is missing."""
        from ragguard.types import StandardSearchResult

        result_dict = {
            "id": "doc1",
            "metadata": {"type": "article"}
        }

        result = StandardSearchResult.from_chromadb(result_dict)

        assert result.id == "doc1"
        assert result.score == 1.0  # 1/(1+0)

    def test_from_chromadb_none_distance(self):
        """Test from_chromadb when distance is None."""
        from ragguard.types import StandardSearchResult

        result_dict = {
            "id": "doc1",
            "distance": None,
            "metadata": {}
        }

        result = StandardSearchResult.from_chromadb(result_dict)

        assert result.score == 0.0  # distance is None

    def test_from_pinecone_basic(self):
        """Test converting Pinecone match to StandardSearchResult."""
        from ragguard.types import StandardSearchResult

        match = MagicMock()
        match.id = "pinecone_vec_1"
        match.score = 0.92
        match.metadata = {"text": "Pinecone document", "source": "web"}
        match.values = [0.5, 0.6, 0.7]

        result = StandardSearchResult.from_pinecone(match)

        assert result.id == "pinecone_vec_1"
        assert result.score == 0.92
        assert result.metadata == {"text": "Pinecone document", "source": "web"}
        assert result.text == "Pinecone document"
        assert result.vector == [0.5, 0.6, 0.7]

    def test_from_pinecone_no_metadata(self):
        """Test from_pinecone when metadata is None."""
        from ragguard.types import StandardSearchResult

        match = MagicMock()
        match.id = "vec1"
        match.score = 0.8
        match.metadata = None
        match.values = None

        result = StandardSearchResult.from_pinecone(match)

        assert result.id == "vec1"
        assert result.metadata == {}
        assert result.text is None
        assert result.vector is None

    def test_from_weaviate_basic(self):
        """Test converting Weaviate object to StandardSearchResult."""
        from ragguard.types import StandardSearchResult

        obj = {
            "text": "Weaviate document",
            "author": "Bob",
            "_additional": {
                "id": "weaviate-uuid-123",
                "certainty": 0.85,
                "vector": [0.1, 0.2]
            }
        }

        result = StandardSearchResult.from_weaviate(obj)

        assert result.id == "weaviate-uuid-123"
        assert result.score == 0.85
        assert "text" in result.metadata
        assert "author" in result.metadata
        assert "_additional" not in result.metadata
        assert result.text == "Weaviate document"
        assert result.vector == [0.1, 0.2]

    def test_from_weaviate_with_distance(self):
        """Test from_weaviate using distance instead of certainty."""
        from ragguard.types import StandardSearchResult

        obj = {
            "content": "Alternative text field",
            "_additional": {
                "id": "wv-123",
                "distance": 0.2
            }
        }

        result = StandardSearchResult.from_weaviate(obj)

        assert result.id == "wv-123"
        assert result.score == 0.2
        assert result.text == "Alternative text field"

    def test_from_weaviate_no_additional(self):
        """Test from_weaviate without _additional field."""
        from ragguard.types import StandardSearchResult

        obj = {
            "id": "fallback-id",
            "text": "Simple doc"
        }

        result = StandardSearchResult.from_weaviate(obj)

        assert result.id == "fallback-id"
        assert result.score == 0.0
        assert result.text == "Simple doc"

    def test_from_weaviate_no_score(self):
        """Test from_weaviate when certainty is None."""
        from ragguard.types import StandardSearchResult

        obj = {
            "text": "Doc",
            "_additional": {
                "id": "id1",
                "certainty": None
            }
        }

        result = StandardSearchResult.from_weaviate(obj)

        assert result.id == "id1"
        assert result.score == 0.0

    def test_from_dict_basic(self):
        """Test from_dict with standard fields."""
        from ragguard.types import StandardSearchResult

        d = {
            "id": "generic-123",
            "score": 0.75,
            "metadata": {"key": "value"},
            "text": "Generic document",
            "vector": [1.0, 2.0, 3.0]
        }

        result = StandardSearchResult.from_dict(d)

        assert result.id == "generic-123"
        assert result.score == 0.75
        assert result.metadata == {"key": "value"}
        assert result.text == "Generic document"
        assert result.vector == [1.0, 2.0, 3.0]

    def test_from_dict_alternate_fields(self):
        """Test from_dict with alternate field names."""
        from ragguard.types import StandardSearchResult

        d = {
            "id": "alt-123",
            "distance": 0.3,  # Instead of score
            "payload": {"attr": "val"},  # Instead of metadata
            "document": "Doc content",  # Instead of text
            "embedding": [0.1, 0.2]  # Instead of vector
        }

        result = StandardSearchResult.from_dict(d)

        assert result.id == "alt-123"
        assert result.score == 0.3
        assert result.metadata == {"attr": "val"}
        assert result.text == "Doc content"
        assert result.vector == [0.1, 0.2]

    def test_from_dict_content_field(self):
        """Test from_dict with 'content' as text field."""
        from ragguard.types import StandardSearchResult

        d = {
            "id": "c123",
            "score": 0.5,
            "content": "Content text"
        }

        result = StandardSearchResult.from_dict(d)

        assert result.text == "Content text"

    def test_from_dict_empty(self):
        """Test from_dict with minimal/empty dict."""
        from ragguard.types import StandardSearchResult

        d = {}

        result = StandardSearchResult.from_dict(d)

        assert result.id == ""
        assert result.score == 0.0
        assert result.metadata == {}
        assert result.text is None
        assert result.vector is None


class TestValidateVectorDimension:
    """Tests for validate_vector_dimension function."""

    def test_valid_dimension(self):
        """Test with valid dimension - should not raise."""
        from ragguard.types import validate_vector_dimension

        # Should not raise
        validate_vector_dimension([0.1, 0.2, 0.3], 3)
        validate_vector_dimension([0.0] * 768, 768)
        validate_vector_dimension([], 0)

    def test_invalid_dimension_too_short(self):
        """Test with vector shorter than expected."""
        from ragguard.types import validate_vector_dimension

        with pytest.raises(ValueError) as exc:
            validate_vector_dimension([0.1, 0.2], 768)

        assert "dimension mismatch" in str(exc.value).lower()
        assert "expected 768" in str(exc.value)
        assert "got 2" in str(exc.value)

    def test_invalid_dimension_too_long(self):
        """Test with vector longer than expected."""
        from ragguard.types import validate_vector_dimension

        with pytest.raises(ValueError) as exc:
            validate_vector_dimension([0.0] * 1024, 768)

        assert "expected 768" in str(exc.value)
        assert "got 1024" in str(exc.value)

    def test_custom_context(self):
        """Test custom context in error message."""
        from ragguard.types import validate_vector_dimension

        with pytest.raises(ValueError) as exc:
            validate_vector_dimension([0.1], 3, context="document embedding")

        assert "document embedding" in str(exc.value)


class TestProtocolTypes:
    """Tests for protocol types."""

    def test_filter_object_protocol(self):
        """Test FilterObject protocol."""
        from ragguard.types import FilterObject

        # Any object should match the protocol (it's empty)
        class MyFilter:
            def __init__(self):
                self.condition = "test"

        f = MyFilter()
        assert isinstance(f, FilterObject)

    def test_vector_database_client_protocol(self):
        """Test VectorDatabaseClient protocol."""
        from ragguard.types import VectorDatabaseClient

        class MockClient:
            def search(self, *args, **kwargs):
                return []

        client = MockClient()
        assert isinstance(client, VectorDatabaseClient)

    def test_has_model_dump_protocol(self):
        """Test HasModelDump protocol."""
        from ragguard.types import HasModelDump

        class MockPydanticModel:
            def model_dump(self):
                return {"field": "value"}

        model = MockPydanticModel()
        assert isinstance(model, HasModelDump)


class TestSearchResultTypedDict:
    """Tests for SearchResult TypedDict."""

    def test_search_result_creation(self):
        """Test creating SearchResult TypedDict."""
        from ragguard.types import SearchResult

        result: SearchResult = {
            "id": "doc1",
            "score": 0.95,
            "payload": {"text": "content"},
            "metadata": {"author": "alice"}
        }

        assert result["id"] == "doc1"
        assert result["score"] == 0.95

    def test_search_result_partial(self):
        """Test SearchResult with partial fields (total=False)."""
        from ragguard.types import SearchResult

        # Only required fields
        result: SearchResult = {
            "id": "doc2"
        }

        assert result["id"] == "doc2"
