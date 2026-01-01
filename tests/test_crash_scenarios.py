"""
Comprehensive crash scenario tests for RAGGuard.

These tests verify the library handles edge cases gracefully and doesn't crash
on invalid input, missing dependencies, or unusual conditions.
"""

import os
import tempfile
import threading
from pathlib import Path

import pytest

# Skip all tests if qdrant_client is not installed
pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from ragguard import AuditLogger, Policy, SecureRetriever, load_policy
from ragguard.exceptions import PolicyError, PolicyValidationError, RetrieverError
from ragguard.policy import PolicyEngine, PolicyParser

# ============================================================================
# User Input Edge Cases
# ============================================================================

def test_none_user_context():
    """Test search with None as user context - should raise validation error."""
    client = QdrantClient(":memory:")
    client.create_collection("test", vectors_config=VectorParams(size=128, distance=Distance.COSINE))

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "public", "allow": {"everyone": True}}],
        "default": "deny"
    })

    retriever = SecureRetriever(client=client, collection="test", policy=policy)

    # Should raise RetrieverError with clear message (not crash)
    with pytest.raises(RetrieverError, match="User context must be a dictionary"):
        retriever.search(query=[0.1]*128, user=None, limit=10)


def test_empty_user_dict():
    """Test search with empty user dictionary."""
    client = QdrantClient(":memory:")
    client.create_collection("test", vectors_config=VectorParams(size=128, distance=Distance.COSINE))

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "test", "allow": {"everyone": True}}],
        "default": "deny"
    })

    retriever = SecureRetriever(client=client, collection="test", policy=policy)

    # Should raise RetrieverError for empty user context (requires 'id' field)
    with pytest.raises(RetrieverError, match="Empty User Context"):
        retriever.search(query=[0.1]*128, user={}, limit=10)


def test_user_with_none_roles():
    """Test user with None roles field."""
    client = QdrantClient(":memory:")
    client.create_collection("test", vectors_config=VectorParams(size=128, distance=Distance.COSINE))

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "test", "allow": {"roles": ["admin"]}}],
        "default": "deny"
    })

    retriever = SecureRetriever(client=client, collection="test", policy=policy)

    # Should not crash
    results = retriever.search(query=[0.1]*128, user={"roles": None}, limit=10)
    assert results is not None


def test_user_missing_required_field():
    """Test user missing a field referenced in policy."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "test", "allow": {"conditions": ["user.department == document.department"]}}],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # User missing 'department' field
    result = engine.evaluate({"id": "alice"}, {"department": "eng"})
    assert result == False  # Should deny, not crash


def test_roles_as_string_not_list():
    """Test user.roles as string instead of list."""
    client = QdrantClient(":memory:")
    client.create_collection("test", vectors_config=VectorParams(size=128, distance=Distance.COSINE))

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "test", "allow": {"roles": ["admin"]}}],
        "default": "deny"
    })

    retriever = SecureRetriever(client=client, collection="test", policy=policy)

    # roles as string should be handled
    results = retriever.search(query=[0.1]*128, user={"roles": "admin"}, limit=10)
    assert results is not None


# ============================================================================
# Type Safety Edge Cases
# ============================================================================

def test_very_large_user_context():
    """Test handling of very large user context (10MB)."""
    client = QdrantClient(":memory:")
    client.create_collection("test", vectors_config=VectorParams(size=128, distance=Distance.COSINE))

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "test", "allow": {"everyone": True}}],
        "default": "deny"
    })

    retriever = SecureRetriever(client=client, collection="test", policy=policy)

    # 10MB of data - should be rejected by validation (exceeds max string length)
    huge_user = {"id": "test", "data": "x" * 10_000_000}
    with pytest.raises(RetrieverError, match="string length .* exceeds maximum"):
        retriever.search(query=[0.1]*128, user=huge_user, limit=10)


def test_circular_reference_in_user():
    """Test user context with circular reference."""
    client = QdrantClient(":memory:")
    client.create_collection("test", vectors_config=VectorParams(size=128, distance=Distance.COSINE))

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "test", "allow": {"everyone": True}}],
        "default": "deny"
    })

    retriever = SecureRetriever(client=client, collection="test", policy=policy)

    circular_user = {"id": "test"}
    circular_user["self"] = circular_user

    # Should detect circular reference and reject (exceeds nesting depth)
    with pytest.raises(RetrieverError, match="exceeds maximum nesting depth"):
        retriever.search(query=[0.1]*128, user=circular_user, limit=10)


def test_unicode_in_user_fields():
    """Test Unicode characters in user context."""
    client = QdrantClient(":memory:")
    client.create_collection("test", vectors_config=VectorParams(size=128, distance=Distance.COSINE))

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "test", "allow": {"everyone": True}}],
        "default": "deny"
    })

    retriever = SecureRetriever(client=client, collection="test", policy=policy)

    unicode_user = {
        "id": "test",
        "name": "测试用户 🎉",
        "department": "مهندس"
    }

    results = retriever.search(query=[0.1]*128, user=unicode_user, limit=10)
    assert results is not None


def test_deep_nesting_in_condition():
    """Test deeply nested field access in conditions."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "test",
            "allow": {"conditions": ["user.a.b.c.d.e.f.g == document.x.y.z"]}
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # 7 levels deep
    user = {"a": {"b": {"c": {"d": {"e": {"f": {"g": "value"}}}}}}}
    doc = {"x": {"y": {"z": "value"}}}

    result = engine.evaluate(user, doc)
    assert result == True


# ============================================================================
# Security Tests
# ============================================================================

def test_sql_injection_in_condition():
    """Test that SQL injection attempts are harmless."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "test", "allow": {"conditions": ["user.id == document.owner"]}}],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # SQL injection attempt
    result = engine.evaluate(
        {"id": "' OR '1'='1"},
        {"owner": "legitimate_user"}
    )

    # Should just do string comparison, not execute SQL
    assert result == False


def test_code_injection_attempt():
    """Test that code injection attempts don't execute."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "test", "allow": {"conditions": ["user.id == document.owner"]}}],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Code injection attempt
    result = engine.evaluate(
        {"id": '__import__("os").system("echo pwned")'},
        {"owner": "user1"}
    )

    # Should be safe string comparison
    assert result == False


# ============================================================================
# Qdrant Edge Cases
# ============================================================================

def test_collection_not_found():
    """Test handling when Qdrant collection doesn't exist."""
    client = QdrantClient(":memory:")
    # Don't create collection

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "test", "allow": {"everyone": True}}],
        "default": "deny"
    })

    retriever = SecureRetriever(client=client, collection="nonexistent", policy=policy)

    # Should raise RetrieverError, not crash
    with pytest.raises(RetrieverError):
        retriever.search(query=[0.1]*128, user={"id": "test"}, limit=10)


def test_wrong_vector_dimension():
    """Test passing vector with wrong dimension."""
    client = QdrantClient(":memory:")
    client.create_collection("test", vectors_config=VectorParams(size=128, distance=Distance.COSINE))

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "test", "allow": {"everyone": True}}],
        "default": "deny"
    })

    retriever = SecureRetriever(client=client, collection="test", policy=policy)

    # Wrong dimension (64 instead of 128)
    with pytest.raises(RetrieverError):
        retriever.search(query=[0.1]*64, user={"id": "test"}, limit=10)


def test_text_query_without_embed_fn():
    """Test that text query without embed function raises clear error."""
    client = QdrantClient(":memory:")
    client.create_collection("test", vectors_config=VectorParams(size=128, distance=Distance.COSINE))

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "test", "allow": {"everyone": True}}],
        "default": "deny"
    })

    retriever = SecureRetriever(client=client, collection="test", policy=policy)

    # Text query without embed_fn
    with pytest.raises(RetrieverError, match="embed_fn"):
        retriever.search(query="text query", user={"id": "test"}, limit=10)


def test_negative_limit():
    """Test negative limit parameter."""
    client = QdrantClient(":memory:")
    client.create_collection("test", vectors_config=VectorParams(size=128, distance=Distance.COSINE))

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "test", "allow": {"everyone": True}}],
        "default": "deny"
    })

    retriever = SecureRetriever(client=client, collection="test", policy=policy)

    # Negative limit - Qdrant will handle this
    results = retriever.search(query=[0.1]*128, user={"id": "test"}, limit=-1)
    assert results is not None


# ============================================================================
# File I/O Edge Cases
# ============================================================================

def test_nonexistent_policy_file():
    """Test loading policy from nonexistent file."""
    with pytest.raises(PolicyError, match="not found"):
        load_policy("/path/that/does/not/exist.yaml")


def test_empty_yaml_file():
    """Test loading empty YAML file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('')
        f.flush()
        temp_path = f.name

    try:
        with pytest.raises(PolicyError, match="empty"):
            load_policy(temp_path)
    finally:
        os.unlink(temp_path)


def test_malformed_yaml():
    """Test loading malformed YAML."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("{{invalid yaml content")
        f.flush()
        temp_path = f.name

    try:
        with pytest.raises(PolicyError, match="parse"):
            load_policy(temp_path)
    finally:
        os.unlink(temp_path)


def test_audit_log_permission_denied():
    """Test audit logger with unwritable path - should handle gracefully."""
    # Try to write to a system directory (will fail)
    logger = AuditLogger(output='file:/etc/ragguard_audit.log')

    # Verify initial counts
    assert logger.failure_count == 0
    assert logger.success_count == 0

    # Should handle permission error gracefully by tracking failure
    logger.log(user={'id': 'test'}, query='test', results_count=0, filter_applied=None)

    # Verify failure was tracked
    assert logger.failure_count == 1
    assert logger.success_count == 0


def test_audit_log_raise_on_failure():
    """Test audit logger raises on failure when configured."""
    logger = AuditLogger(output='file:/etc/ragguard_audit.log', raise_on_failure=True)

    # Should raise RuntimeError on failure
    with pytest.raises(RuntimeError, match="Audit logging failed"):
        logger.log(user={'id': 'test'}, query='test', results_count=0, filter_applied=None)


def test_audit_log_on_failure_callback():
    """Test audit logger calls on_failure callback."""
    failures = []

    def on_failure(error, entry):
        failures.append((error, entry))

    logger = AuditLogger(output='file:/etc/ragguard_audit.log', on_failure=on_failure)

    logger.log(user={'id': 'test'}, query='test', results_count=0, filter_applied=None)

    # Verify callback was called
    assert len(failures) == 1
    assert 'Permission denied' in str(failures[0][0]) or 'protected system directory' in str(failures[0][0])
    assert failures[0][1]['user_id'] == 'test'


# ============================================================================
# Policy Validation Edge Cases
# ============================================================================

def test_policy_empty_rules():
    """Test that empty rules list is rejected."""
    # Pydantic raises its own ValidationError, not PolicyValidationError
    with pytest.raises(Exception) as exc_info:
        Policy.from_dict({
            "version": "1",
            "rules": [],
            "default": "deny"
        })
    assert "at least one rule" in str(exc_info.value).lower()


def test_policy_invalid_default():
    """Test that invalid default value is rejected."""
    # Pydantic raises its own ValidationError, not PolicyValidationError
    with pytest.raises(Exception) as exc_info:
        Policy.from_dict({
            "version": "1",
            "rules": [{"name": "test", "allow": {"everyone": True}}],
            "default": "maybe"  # Invalid
        })
    assert "deny" in str(exc_info.value).lower() or "allow" in str(exc_info.value).lower()


def test_policy_invalid_condition_operator():
    """Test that invalid condition operator is rejected at policy creation time."""
    # Policy validation should reject invalid operators
    with pytest.raises(Exception) as exc_info:
        Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "test",
                "allow": {"conditions": ["user.id <> document.id"]}  # <> operator invalid
            }],
            "default": "deny"
        })
    # Should mention valid operators in error message
    error_msg = str(exc_info.value).lower()
    assert "<>" in error_msg or "operator" in error_msg or "==" in error_msg


# ============================================================================
# Concurrency Tests
# ============================================================================

def test_concurrent_searches():
    """Test thread-safety of concurrent searches."""
    client = QdrantClient(":memory:")
    client.create_collection("test", vectors_config=VectorParams(size=128, distance=Distance.COSINE))

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "test", "allow": {"everyone": True}}],
        "default": "deny"
    })

    retriever = SecureRetriever(client=client, collection="test", policy=policy)

    errors = []

    def search_thread():
        try:
            for _ in range(10):
                retriever.search(query=[0.1]*128, user={"id": "test"}, limit=10)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=search_thread) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent searches failed with errors: {errors}"


def test_concurrent_audit_logging():
    """Test thread-safety of concurrent audit logging."""
    client = QdrantClient(":memory:")
    client.create_collection("test", vectors_config=VectorParams(size=128, distance=Distance.COSINE))

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "test", "allow": {"everyone": True}}],
        "default": "deny"
    })

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl')
    temp_file.close()

    try:
        logger = AuditLogger(output=f'file:{temp_file.name}')
        retriever = SecureRetriever(
            client=client,
            collection="test",
            policy=policy,
            audit_logger=logger
        )

        errors = []

        def search_with_logging():
            try:
                for _ in range(10):
                    retriever.search(query=[0.1]*128, user={"id": "test"}, limit=10)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=search_with_logging) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent logging failed with errors: {errors}"
    finally:
        os.unlink(temp_file.name)


# ============================================================================
# Custom Filter Edge Cases
# ============================================================================

def test_custom_filter_returns_none():
    """Test custom filter builder returning None."""
    from ragguard.filters.custom import LambdaFilterBuilder

    client = QdrantClient(":memory:")
    client.create_collection("test", vectors_config=VectorParams(size=128, distance=Distance.COSINE))

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "test", "allow": {"everyone": True}}],
        "default": "deny"
    })

    def filter_returns_none(policy, user):
        return None

    builder = LambdaFilterBuilder(qdrant=filter_returns_none)
    retriever = SecureRetriever(
        client=client,
        collection="test",
        policy=policy,
        custom_filter_builder=builder
    )

    # Should handle None gracefully
    results = retriever.search(query=[0.1]*128, user={"id": "test"}, limit=10)
    assert results is not None


def test_custom_filter_raises_exception():
    """Test custom filter builder that raises exception."""
    from ragguard.filters.custom import LambdaFilterBuilder

    client = QdrantClient(":memory:")
    client.create_collection("test", vectors_config=VectorParams(size=128, distance=Distance.COSINE))

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "test", "allow": {"everyone": True}}],
        "default": "deny"
    })

    def bad_filter(policy, user):
        raise ValueError("Filter crashed!")

    builder = LambdaFilterBuilder(qdrant=bad_filter)
    retriever = SecureRetriever(
        client=client,
        collection="test",
        policy=policy,
        custom_filter_builder=builder
    )

    # Should wrap in RetrieverError
    with pytest.raises(RetrieverError):
        retriever.search(query=[0.1]*128, user={"id": "test"}, limit=10)
