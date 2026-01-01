# Production Features

RAGGuard includes enterprise-grade features for production deployments.

## Input Validation

Prevent DOS and injection attacks:

```python
from ragguard import ChromaDBSecureRetriever, ValidationConfig

retriever = ChromaDBSecureRetriever(
    collection=collection,
    policy=policy,
    enable_validation=True,
    validation_config=ValidationConfig(
        max_dict_size=100,          # Prevent DOS via large payloads
        max_string_length=10000,    # Limit string sizes
        max_nesting_depth=10,       # Prevent deeply nested objects
        strict_field_names=True     # Block injection attempts
    )
)
```

**Protection against:**
- DOS attacks via deeply nested objects or large payloads
- Field name injection (SQL, MongoDB, etc.)
- Type confusion attacks

## Retry Logic

Automatic retry for transient failures with exponential backoff:

```python
from ragguard import ChromaDBSecureRetriever, RetryConfig

retriever = ChromaDBSecureRetriever(
    collection=collection,
    policy=policy,
    enable_retry=True,
    retry_config=RetryConfig(
        max_retries=3,
        initial_delay=0.1,
        max_delay=10.0,
        exponential_base=2,
        jitter=True  # Prevent thundering herd
    )
)
```

## Health Checks

Built-in health checks for all backends:

```python
health = retriever.health_check()

if health["healthy"]:
    print(f"Healthy - {health['backend']}: {health['collection']}")
else:
    print(f"Unhealthy: {health['error']}")

# Response:
# {
#     "healthy": True,
#     "backend": "qdrant",
#     "collection": "documents",
#     "details": {"vectors_count": 10000, "status": "green"}
# }
```

### Kubernetes Probes

```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/health/liveness')
def liveness():
    health = retriever.health_check()
    return jsonify(health), 200 if health["healthy"] else 500

@app.route('/health/readiness')
def readiness():
    health = retriever.health_check()
    return jsonify(health), 200 if health["healthy"] else 503
```

## Structured Logging

Production-ready JSON logging:

```python
from ragguard import get_logger, add_log_context, configure_logging

# Configure once at startup
configure_logging(level="INFO", format="json")

# Add request context
with add_log_context(request_id="req-123", user_id="alice"):
    results = retriever.search(query, user, limit=10)
    # All logs include request_id and user_id
```

**Output:**

```json
{
  "timestamp": "2025-01-15T10:30:45.123456+00:00",
  "level": "INFO",
  "message": "Search completed",
  "request_id": "req-123",
  "user_id": "alice",
  "backend": "chromadb",
  "results_count": 10
}
```

## Async Support

Native async/await for high-performance applications:

```python
from ragguard import AsyncQdrantSecureRetriever

async def search_handler(query: str, user: dict):
    async with AsyncQdrantSecureRetriever(client, "docs", policy) as retriever:
        results = await retriever.search(query=query, user=user, limit=10)
    return results
```

**Supported async backends:**
- AsyncQdrantSecureRetriever
- AsyncChromaDBSecureRetriever
- AsyncPineconeSecureRetriever
- AsyncWeaviateSecureRetriever
- AsyncPgvectorSecureRetriever
- AsyncFAISSSecureRetriever

## Connection Pooling

Optimized connection management for pgvector:

```python
from ragguard import PgvectorConnectionPool, PgvectorSecureRetriever

pool = PgvectorConnectionPool(
    dsn="postgresql://localhost/mydb",
    min_size=2,
    max_size=10,
    timeout=30
)

retriever = PgvectorSecureRetriever(
    connection=pool,
    table="documents",
    policy=policy
)
```

## Audit Logging

Complete audit trail for compliance:

```python
from ragguard import AuditLogger

audit_logger = AuditLogger(
    output="audit.log",
    include_filters=True,
    include_results=False  # Privacy
)

retriever = ChromaDBSecureRetriever(
    collection=collection,
    policy=policy,
    audit_logger=audit_logger
)
```

**Audit log entry:**

```json
{
  "timestamp": "2025-01-15T10:30:45.123456",
  "user_id": "alice",
  "query": "Q3 revenue",
  "results_count": 10,
  "filter_applied": {"department": {"$eq": "finance"}},
  "backend": "chromadb"
}
```

## Circuit Breaker

Prevent cascade failures:

```python
from ragguard import CircuitBreakerConfig

retriever = ChromaDBSecureRetriever(
    collection=collection,
    policy=policy,
    circuit_breaker_config=CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=30
    )
)
```
