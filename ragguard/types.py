"""
Type definitions and protocols for RAGGuard.

This module provides type aliases and Protocol definitions to improve type safety
throughout the codebase while maintaining compatibility with optional dependencies.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Optional, Protocol, TypeVar, Union, runtime_checkable

from typing_extensions import TypeAlias, TypedDict

# =============================================================================
# User and Document Context Types
# =============================================================================

UserContext: TypeAlias = dict[str, Any]
"""
User context dictionary containing user attributes for permission evaluation.
Should typically include 'id' and may include 'roles', 'department', etc.
"""

DocumentContext: TypeAlias = dict[str, Any]
"""
Document metadata dictionary containing attributes for permission evaluation.
May include 'owner', 'department', 'visibility', etc.
"""


# =============================================================================
# Filter Object Types
# =============================================================================

@runtime_checkable
class FilterObject(Protocol):
    """
    Protocol for database filter objects.

    Database filters can be any object that the underlying vector database
    accepts as a filter condition. This protocol provides type safety without
    requiring specific filter implementations.
    """
    pass


FilterType: TypeAlias = Union[FilterObject, dict[str, Any], object, None]
"""
Type alias for database-specific filter objects.

Different backends return different filter types:
- Qdrant: qdrant_client.models.Filter
- Weaviate: dict
- Pinecone: dict
- PgVector: str (SQL WHERE clause)
- ChromaDB: dict
- FAISS: None (filters in metadata)
"""

# Type variable for generic filter types
T = TypeVar('T')


class FilterResultType(Enum):
    """
    Explicit semantics for filter builder results.

    SECURITY: Using explicit result types prevents ambiguous filter results
    that could lead to unintended access (e.g., None meaning "allow all"
    vs None meaning "error occurred").
    """
    ALLOW_ALL = "allow_all"      # No filter needed - allow all documents
    DENY_ALL = "deny_all"        # Match nothing - deny all access
    CONDITIONAL = "conditional"  # Normal filter with conditions


@dataclass
class FilterResult(Generic[T]):
    """
    Standardized filter result with explicit semantics.

    SECURITY: This class makes filter semantics explicit to prevent
    silent failures where ambiguous return values (None, empty dict)
    could lead to unintended access.

    Example:
        >>> result = FilterResult.deny_all()
        >>> if result.result_type == FilterResultType.DENY_ALL:
        ...     # No need to query the database - access is denied
        ...     return []

        >>> result = FilterResult.conditional({"department": "engineering"})
        >>> native_filter = result.filter  # Use in database query
    """
    result_type: FilterResultType
    filter: Optional[T]
    reason: Optional[str] = None

    @classmethod
    def allow_all(cls) -> 'FilterResult[T]':
        """Create a result that allows all documents (no filter needed)."""
        return cls(result_type=FilterResultType.ALLOW_ALL, filter=None)

    @classmethod
    def deny_all(cls, reason: Optional[str] = None) -> 'FilterResult[T]':
        """Create a result that denies all access."""
        return cls(
            result_type=FilterResultType.DENY_ALL,
            filter=None,
            reason=reason or "No matching rules grant access"
        )

    @classmethod
    def conditional(cls, filter_obj: T) -> 'FilterResult[T]':
        """Create a result with a conditional filter."""
        return cls(result_type=FilterResultType.CONDITIONAL, filter=filter_obj)

    @property
    def is_deny_all(self) -> bool:
        """Check if this result denies all access."""
        return self.result_type == FilterResultType.DENY_ALL

    @property
    def is_allow_all(self) -> bool:
        """Check if this result allows all access (no filter)."""
        return self.result_type == FilterResultType.ALLOW_ALL

    @property
    def is_conditional(self) -> bool:
        """Check if this result has a conditional filter."""
        return self.result_type == FilterResultType.CONDITIONAL


# Backend-specific deny-all filter constants
# These are the actual filter values that each backend interprets as "match nothing"
DENY_ALL_FILTERS: dict[str, Any] = {
    "qdrant": None,  # Qdrant: use must_not with always-true condition
    "pgvector": ("WHERE FALSE", []),
    "chromadb": {"$and": [{"__deny_all__": {"$eq": "__never_match__"}}]},
    "pinecone": {"__deny_all__": {"$eq": "__never_match__"}},
    "weaviate": {"operator": "And", "operands": []},  # Empty AND matches nothing
    "milvus": "1 == 0",
    "elasticsearch": {"bool": {"must": [{"term": {"__deny_all__": "__never_match__"}}]}},
    "azure_search": "__deny_all__ eq '__never_match__'",
}


# =============================================================================
# Database Client Protocols
# =============================================================================

@runtime_checkable
class VectorDatabaseClient(Protocol):
    """
    Protocol for vector database clients.

    This protocol defines the minimal interface expected from vector database
    clients. Actual clients may have additional methods and attributes.
    """

    def search(self, *args: Any, **kwargs: Any) -> Any:
        """Perform a vector similarity search."""
        ...


@runtime_checkable
class HasModelDump(Protocol):
    """Protocol for Pydantic models with model_dump method."""

    def model_dump(self) -> dict[str, Any]:
        """Dump model to dictionary."""
        ...


# =============================================================================
# Query and Result Types
# =============================================================================

class SearchResult(TypedDict, total=False):
    """
    Type definition for search results returned by retrievers.

    Note: Actual return types vary by backend. This is a common subset.
    """
    id: str
    score: float
    payload: dict[str, Any]
    metadata: dict[str, Any]
    text: str
    embedding: list[float]


@dataclass
class StandardSearchResult:
    """
    Standardized search result that works across all backends.

    Use this class when you need a consistent interface regardless of backend.
    All backends can convert their native results to this format.

    Attributes:
        id: Unique identifier for the result
        score: Similarity score (higher = more similar, normalized 0-1 when possible)
        metadata: Document metadata dictionary
        text: Optional document text content
        vector: Optional embedding vector

    Example:
        ```python
        # Convert backend results to standard format
        results = retriever.search(query, user)
        standard_results = [StandardSearchResult.from_qdrant(r) for r in results]

        # Now you can use consistent interface
        for result in standard_results:
            print(f"{result.id}: {result.score:.3f}")
            print(f"  metadata: {result.metadata}")
        ```
    """
    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    text: Optional[str] = None
    vector: Optional[list[float]] = None

    @classmethod
    def from_qdrant(cls, point: Any) -> "StandardSearchResult":
        """Convert Qdrant ScoredPoint to StandardSearchResult."""
        return cls(
            id=str(point.id),
            score=point.score,
            metadata=point.payload or {},
            text=point.payload.get("text") if point.payload else None,
            vector=point.vector if hasattr(point, 'vector') else None
        )

    @classmethod
    def from_chromadb(cls, result: dict[str, Any]) -> "StandardSearchResult":
        """Convert ChromaDB result dict to StandardSearchResult."""
        # ChromaDB returns distance, convert to similarity (1 - distance for L2)
        distance = result.get("distance", 0.0)
        # Normalize: assume L2 distance, convert to 0-1 similarity
        score = 1.0 / (1.0 + distance) if distance is not None else 0.0
        return cls(
            id=str(result.get("id", "")),
            score=score,
            metadata=result.get("metadata", {}),
            text=result.get("document"),
            vector=None
        )

    @classmethod
    def from_pinecone(cls, match: Any) -> "StandardSearchResult":
        """Convert Pinecone match to StandardSearchResult."""
        return cls(
            id=str(match.id),
            score=match.score,
            metadata=match.metadata or {},
            text=match.metadata.get("text") if match.metadata else None,
            vector=match.values if hasattr(match, 'values') else None
        )

    @classmethod
    def from_weaviate(cls, obj: dict[str, Any]) -> "StandardSearchResult":
        """Convert Weaviate object to StandardSearchResult."""
        # Weaviate includes _additional for score info
        additional = obj.get("_additional", {})
        score = additional.get("certainty", 0.0) or additional.get("distance", 0.0)

        # Extract id
        obj_id = additional.get("id", obj.get("id", ""))

        # Metadata is the object itself minus _additional
        metadata = {k: v for k, v in obj.items() if k != "_additional"}

        return cls(
            id=str(obj_id),
            score=float(score) if score else 0.0,
            metadata=metadata,
            text=obj.get("text") or obj.get("content"),
            vector=additional.get("vector")
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StandardSearchResult":
        """Convert generic dict to StandardSearchResult."""
        return cls(
            id=str(d.get("id", "")),
            score=float(d.get("score", d.get("distance", 0.0))),
            metadata=d.get("metadata", d.get("payload", {})),
            text=d.get("text", d.get("document", d.get("content"))),
            vector=d.get("vector", d.get("embedding"))
        )


EmbeddingVector: TypeAlias = list[float]
"""Type alias for embedding vectors."""


def validate_vector_dimension(
    vector: list[float],
    expected_dimension: int,
    context: str = "query"
) -> None:
    """
    Validate that a vector has the expected dimension.

    Args:
        vector: The vector to validate
        expected_dimension: Expected number of dimensions
        context: Context for error message (e.g., "query", "document")

    Raises:
        ValueError: If vector dimension doesn't match expected

    Example:
        >>> validate_vector_dimension([0.1, 0.2, 0.3], 3)  # OK
        >>> validate_vector_dimension([0.1, 0.2], 768)  # Raises ValueError
    """
    actual_dimension = len(vector)
    if actual_dimension != expected_dimension:
        raise ValueError(
            f"Vector dimension mismatch for {context}: "
            f"expected {expected_dimension}, got {actual_dimension}. "
            f"Ensure your embedding model produces {expected_dimension}-dimensional vectors."
        )


# =============================================================================
# Cache Value Types
# =============================================================================

CachedFilter: TypeAlias = Union[FilterObject, object]
"""
Type for cached filter objects.

Filters are cached as opaque objects since their structure varies by backend.
The cache doesn't need to inspect filter internals, just store and retrieve them.
"""


# =============================================================================
# Policy-Related Types
# =============================================================================

PolicyDict: TypeAlias = dict[str, Any]
"""
Dictionary representation of a policy before validation.

Expected structure:
{
    "version": "1",
    "rules": [...],
    "default": "deny" | "allow"
}
"""


# =============================================================================
# Backend Types
# =============================================================================

BackendName: TypeAlias = str
"""
Type alias for backend names.

Common values: "qdrant", "pgvector", "weaviate", "pinecone", "chromadb",
"faiss", "elasticsearch", "opensearch", "azure_search", "milvus"
"""
