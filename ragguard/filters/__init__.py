"""
Filter builders for database-specific permission filters.

Each filter builder converts RAGGuard policies to native database filter format.
"""

from .builder import (
    to_azure_search_filter,
    to_chromadb_filter,
    to_elasticsearch_filter,
    to_milvus_filter,
    to_pgvector_filter,
    to_pinecone_filter,
    to_qdrant_filter,
    to_weaviate_filter,
)
from .builder_base import (
    ChromaDBFilterBuilder,
    DictFilterBuilder,
    FilterBuilderBase,
    PineconeFilterBuilder,
)

__all__ = [
    # Filter functions
    "to_azure_search_filter",
    "to_chromadb_filter",
    "to_elasticsearch_filter",
    "to_milvus_filter",
    "to_pgvector_filter",
    "to_pinecone_filter",
    "to_qdrant_filter",
    "to_weaviate_filter",
    # Base classes for custom backends
    "FilterBuilderBase",
    "DictFilterBuilder",
    "PineconeFilterBuilder",
    "ChromaDBFilterBuilder",
]
