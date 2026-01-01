"""
Constants used throughout the RAGGuard library.

This module centralizes magic numbers, string constants, and configuration
defaults to improve maintainability and prevent inconsistencies.
"""

from typing import Final

# =============================================================================
# Field Prefixes
# =============================================================================

USER_PREFIX: Final[str] = "user."
DOCUMENT_PREFIX: Final[str] = "document."

# Length of prefixes (for efficient slicing)
USER_PREFIX_LEN: Final[int] = len(USER_PREFIX)  # 5
DOCUMENT_PREFIX_LEN: Final[int] = len(DOCUMENT_PREFIX)  # 9

# =============================================================================
# Security Limits (DoS Protection)
# =============================================================================

# Maximum lengths for various inputs
MAX_BACKEND_NAME_LENGTH: Final[int] = 100
MAX_CONDITION_STRING_LENGTH: Final[int] = 10000
MAX_FIELD_NAME_LENGTH: Final[int] = 500
MAX_STRING_VALUE_LENGTH: Final[int] = 10000

# Maximum nesting/complexity limits
MAX_EXPRESSION_DEPTH: Final[int] = 50
MAX_CONDITIONS_PER_RULE: Final[int] = 100
MAX_RULES_PER_POLICY: Final[int] = 1000
MAX_OR_CLAUSES: Final[int] = 100

# Number safety limits
MAX_SAFE_INTEGER: Final[int] = 9007199254740991  # JavaScript compatibility limit
MAX_FLOAT_VALUE: Final[float] = 1e308  # Near Python float max

# =============================================================================
# Cache Configuration Defaults
# =============================================================================

DEFAULT_CACHE_SIZE: Final[int] = 1000
DEFAULT_CACHE_TTL_SECONDS: Final[int] = 3600  # 1 hour

# =============================================================================
# Retry Configuration Defaults
# =============================================================================

DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_INITIAL_DELAY: Final[float] = 0.1  # seconds
DEFAULT_MAX_DELAY: Final[float] = 10.0  # seconds
DEFAULT_EXPONENTIAL_BASE: Final[float] = 2.0

# =============================================================================
# Validation Defaults
# =============================================================================

DEFAULT_MAX_DICT_SIZE: Final[int] = 100
DEFAULT_MAX_LIST_SIZE: Final[int] = 1000
DEFAULT_MAX_NESTING_DEPTH: Final[int] = 10

# =============================================================================
# Internal Sentinel Values
# =============================================================================

# Used by filter builders to create deny-all filters
DENY_ALL_FIELD: Final[str] = "__ragguard_deny_all__"
DENY_ALL_VALUE: Final[str] = "__ragguard_deny_all_sentinel__"


class _SkipRuleSentinel:
    """
    Sentinel value indicating a rule should be skipped for this user.

    This is different from None, which means "match all documents".
    Used by filter builders to differentiate between:
    - User doesn't satisfy this rule -> SKIP_RULE (skip to next rule)
    - User satisfies rule with no restrictions -> None (matches all docs)
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "SKIP_RULE"


# Singleton instance for use in filter builders
SKIP_RULE: Final[_SkipRuleSentinel] = _SkipRuleSentinel()

# =============================================================================
# Operator Constants
# =============================================================================

# Comparison operators
OP_EQUALS: Final[str] = "=="
OP_NOT_EQUALS: Final[str] = "!="
OP_GREATER_THAN: Final[str] = ">"
OP_GREATER_EQUAL: Final[str] = ">="
OP_LESS_THAN: Final[str] = "<"
OP_LESS_EQUAL: Final[str] = "<="

# Membership operators
OP_IN: Final[str] = " in "
OP_NOT_IN: Final[str] = " not in "

# Existence operators
OP_EXISTS: Final[str] = " exists"
OP_NOT_EXISTS: Final[str] = " not exists"

# Logical operators
OP_AND: Final[str] = " AND "
OP_OR: Final[str] = " OR "

# All operators for parsing (order matters - longer operators first)
ALL_OPERATORS: Final[tuple[str, ...]] = (
    OP_NOT_EQUALS,
    OP_GREATER_EQUAL,
    OP_LESS_EQUAL,
    OP_EQUALS,
    OP_GREATER_THAN,
    OP_LESS_THAN,
    OP_NOT_IN,
    OP_IN,
    OP_NOT_EXISTS,
    OP_EXISTS,
)

# Operators that compare two values
BINARY_OPERATORS: Final[tuple[str, ...]] = (
    OP_NOT_EQUALS,
    OP_GREATER_EQUAL,
    OP_LESS_EQUAL,
    OP_EQUALS,
    OP_GREATER_THAN,
    OP_LESS_THAN,
    OP_NOT_IN,
    OP_IN,
)

# Operators that check existence only
UNARY_OPERATORS: Final[tuple[str, ...]] = (
    OP_EXISTS,
    OP_NOT_EXISTS,
)

# =============================================================================
# Backend Names
# =============================================================================

BACKEND_QDRANT: Final[str] = "qdrant"
BACKEND_CHROMADB: Final[str] = "chromadb"
BACKEND_PGVECTOR: Final[str] = "pgvector"
BACKEND_PINECONE: Final[str] = "pinecone"
BACKEND_WEAVIATE: Final[str] = "weaviate"
BACKEND_FAISS: Final[str] = "faiss"
BACKEND_MILVUS: Final[str] = "milvus"
BACKEND_ELASTICSEARCH: Final[str] = "elasticsearch"
BACKEND_OPENSEARCH: Final[str] = "opensearch"
BACKEND_AZURE_SEARCH: Final[str] = "azure_search"

ALL_BACKENDS: Final[tuple[str, ...]] = (
    BACKEND_QDRANT,
    BACKEND_CHROMADB,
    BACKEND_PGVECTOR,
    BACKEND_PINECONE,
    BACKEND_WEAVIATE,
    BACKEND_FAISS,
    BACKEND_MILVUS,
    BACKEND_ELASTICSEARCH,
    BACKEND_OPENSEARCH,
    BACKEND_AZURE_SEARCH,
)

# Backends that support native filtering
NATIVE_FILTER_BACKENDS: Final[tuple[str, ...]] = (
    BACKEND_QDRANT,
    BACKEND_CHROMADB,
    BACKEND_PGVECTOR,
    BACKEND_PINECONE,
    BACKEND_WEAVIATE,
    BACKEND_MILVUS,
    BACKEND_ELASTICSEARCH,
    BACKEND_OPENSEARCH,
    BACKEND_AZURE_SEARCH,
)

# Backends that use post-filtering
POST_FILTER_BACKENDS: Final[tuple[str, ...]] = (
    BACKEND_FAISS,
)
