"""
RAGGuard: Permission-aware retrieval for RAG applications.

RAGGuard wraps vector database queries and enforces document-level permissions
by injecting permission filters INTO the vector search, not after.
"""

from .audit import AuditLogger
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitBreakerStats,
    CircuitState,
    get_circuit_breaker,
    reset_all_circuit_breakers,
)
from .config import SecureRetrieverConfig
from .connection_pool import (
    ConnectionPoolProtocol,
    ManagedConnection,
    PgvectorConnectionPool,
)
from .errors import (
    ErrorContext,
    connection_error,
    missing_dependency_error,
    unsupported_backend_error,
    validation_error,
)
from .exceptions import (
    NON_RETRYABLE_EXCEPTIONS,
    RETRYABLE_EXCEPTIONS,
    AuditLogError,
    BackendError,
    ConfigurationError,
    FilterBuildError,
    HealthCheckError,
    PolicyError,
    PolicyEvaluationError,
    PolicyParseError,
    PolicyValidationError,
    QuotaExceededError,
    RAGGuardError,
    RateLimitError,
    RetrieverConnectionError,
    RetrieverError,
    RetrieverPermissionError,
    RetrieverTimeoutError,
    UnsupportedConditionError,
)
from .health import (
    HealthCheckManager,
    create_fastapi_health_endpoints,
    create_flask_health_endpoints,
)
from .logging import (
    add_log_context,
    clear_log_context,
    configure_logging,
    generate_correlation_id,
    get_log_context,
    get_logger,
    request_context,
    set_log_level,
)
from .policy import (
    ConditionEvaluation,
    Policy,
    PolicyEngine,
    PolicyParser,
    QueryExplainer,
    QueryExplanation,
    RuleEvaluation,
    load_policy,
)
from .policy.validator import (
    PolicyValidator,
    ValidationIssue,
    ValidationLevel,
    print_validation_issues,
    validate_policy,
)
from .retrievers import (
    ArangoDBSecureRetriever,
    AzureCognitiveSearchSecureRetriever,
    AzureSearchSecureRetriever,
    # Graph databases
    BaseGraphRetriever,
    ChromaDBSecureRetriever,
    ElasticsearchSecureRetriever,
    FAISSSecureRetriever,
    MilvusSecureRetriever,
    Neo4jSecureRetriever,
    NeptuneSecureRetriever,
    OpenSearchSecureRetriever,
    PgvectorSecureRetriever,
    PineconeSecureRetriever,
    # Vector databases
    QdrantSecureRetriever,
    TigerGraphSecureRetriever,
    WeaviateSecureRetriever,
    ZillizSecureRetriever,
)
from .retry import RetryConfig

# Testing framework moved to ragguard_enterprise.testing
# For policy simulation and test frameworks, install ragguard-enterprise
from .types import (
    BackendName,
    DocumentContext,
    EmbeddingVector,
    FilterType,
    PolicyDict,
    UserContext,
)
from .utils import (
    get_nested_value,
    secure_compare,
    secure_contains,
    strip_document_prefix,
    strip_user_prefix,
)
from .validation import (
    InputValidator,
    ValidationConfig,
    validate_document,
    validate_user,
)

# Metrics moved to ragguard_enterprise.metrics
# For Prometheus/JSON metrics export, install ragguard-enterprise
_metrics_available = False

# Async retrievers (optional, requires async database clients)
try:
    from .retrievers_async import (
        AsyncChromaDBSecureRetriever,
        AsyncFAISSSecureRetriever,
        AsyncPgvectorSecureRetriever,
        AsyncPineconeSecureRetriever,
        AsyncQdrantSecureRetriever,
        AsyncWeaviateSecureRetriever,
        batch_search_async,
        multi_user_search_async,
        run_sync_retriever_async,
    )
    _async_available = True
except ImportError:
    _async_available = False

# Convenient alias (defaults to Qdrant)
SecureRetriever = QdrantSecureRetriever

__version__ = "0.3.0"

__all__ = [
    "Policy",
    "PolicyParser",
    "PolicyEngine",
    "load_policy",
    "QueryExplainer",
    "QueryExplanation",
    "RuleEvaluation",
    "ConditionEvaluation",
    # Testing framework moved to ragguard_enterprise.testing
    "SecureRetriever",
    # Vector databases
    "QdrantSecureRetriever",
    "PgvectorSecureRetriever",
    "WeaviateSecureRetriever",
    "PineconeSecureRetriever",
    "ChromaDBSecureRetriever",
    "FAISSSecureRetriever",
    "MilvusSecureRetriever",
    "ZillizSecureRetriever",
    "ElasticsearchSecureRetriever",
    "OpenSearchSecureRetriever",
    "AzureSearchSecureRetriever",
    "AzureCognitiveSearchSecureRetriever",
    # Graph databases
    "BaseGraphRetriever",
    "Neo4jSecureRetriever",
    "NeptuneSecureRetriever",
    "TigerGraphSecureRetriever",
    "ArangoDBSecureRetriever",
    "AuditLogger",
    "RetryConfig",
    "SecureRetrieverConfig",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpen",
    "CircuitBreakerStats",
    "CircuitState",
    "get_circuit_breaker",
    "reset_all_circuit_breakers",
    "PgvectorConnectionPool",
    "ConnectionPoolProtocol",
    "ManagedConnection",
    "ValidationConfig",
    "InputValidator",
    "validate_user",
    "validate_document",
    "get_logger",
    "add_log_context",
    "set_log_level",
    "configure_logging",
    "get_log_context",
    "clear_log_context",
    "generate_correlation_id",
    "request_context",
    "HealthCheckManager",
    "create_flask_health_endpoints",
    "create_fastapi_health_endpoints",
    "ErrorContext",
    "unsupported_backend_error",
    "missing_dependency_error",
    "validation_error",
    "connection_error",
    "PolicyValidator",
    "ValidationLevel",
    "ValidationIssue",
    "validate_policy",
    "print_validation_issues",
    "UserContext",
    "DocumentContext",
    "FilterType",
    "EmbeddingVector",
    "PolicyDict",
    "BackendName",
    # Utils
    "get_nested_value",
    "strip_user_prefix",
    "strip_document_prefix",
    # Security utilities
    "secure_compare",
    "secure_contains",
    # Exceptions
    "RAGGuardError",
    "PolicyError",
    "PolicyParseError",
    "PolicyValidationError",
    "PolicyEvaluationError",
    "FilterBuildError",
    "UnsupportedConditionError",
    "RetrieverError",
    "RetrieverConnectionError",
    "RetrieverTimeoutError",
    "HealthCheckError",
    "BackendError",
    "RateLimitError",
    "QuotaExceededError",
    "RetrieverPermissionError",
    "ConfigurationError",
    "AuditLogError",
    "RETRYABLE_EXCEPTIONS",
    "NON_RETRYABLE_EXCEPTIONS",
]

# Metrics exports moved to ragguard_enterprise.metrics

# Add async exports if available
if _async_available:
    __all__.extend([
        "AsyncChromaDBSecureRetriever",
        "AsyncFAISSSecureRetriever",
        "AsyncPgvectorSecureRetriever",
        "AsyncPineconeSecureRetriever",
        "AsyncQdrantSecureRetriever",
        "AsyncWeaviateSecureRetriever",
        "batch_search_async",
        "multi_user_search_async",
        "run_sync_retriever_async",
    ])
