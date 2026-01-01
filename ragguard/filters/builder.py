"""
Filter builders for converting policies to database-specific filters.

This module re-exports filter builders from backend-specific modules.
Each `to_<backend>_filter()` function converts a RAGGuard policy and user
context into the backend's native filter format.

For direct backend imports:
    from ragguard.filters.backends import to_qdrant_filter
    from ragguard.filters.backends.pgvector import to_pgvector_filter
"""

# Re-export all filter builders from backends for backwards compatibility
from .backends.azure_search import to_azure_search_filter
from .backends.chromadb import to_chromadb_filter
from .backends.elasticsearch import to_elasticsearch_filter
from .backends.milvus import to_milvus_filter
from .backends.pgvector import to_pgvector_filter
from .backends.pinecone import to_pinecone_filter
from .backends.qdrant import to_qdrant_filter
from .backends.weaviate import to_weaviate_filter
from .base import (
    get_nested_value as _get_nested_value,
)
from .base import (
    parse_list_literal as _parse_list_literal,
)
from .base import (
    parse_literal_value as _parse_literal_value,
)

# Re-export shared utilities for backwards compatibility
from .base import (
    user_satisfies_allow as _user_satisfies_allow,
)
from .base import (
    validate_sql_identifier as _validate_sql_identifier,
)

__all__ = [
    # Main filter builders
    "to_qdrant_filter",
    "to_pgvector_filter",
    "to_weaviate_filter",
    "to_pinecone_filter",
    "to_chromadb_filter",
    "to_milvus_filter",
    "to_elasticsearch_filter",
    "to_azure_search_filter",
    # Legacy internal functions (for backwards compatibility)
    "_user_satisfies_allow",
    "_get_nested_value",
    "_parse_literal_value",
    "_parse_list_literal",
    "_validate_sql_identifier",
]
