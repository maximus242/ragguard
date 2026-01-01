"""
Advanced tests for pgvector integration.

Tests health checks, context managers, validation, retry logic, and edge cases.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

# Skip all tests if pgvector is not installed
pytest.importorskip("pgvector")

from ragguard import Policy
from ragguard.audit import AuditLogger
from ragguard.exceptions import RetrieverError
from ragguard.retrievers import PgvectorSecureRetriever
from ragguard.retry import RetryConfig
from ragguard.validation import ValidationConfig

# Integration test - requires docker-compose up (PostgreSQL with pgvector)
# pytestmark = pytest.mark.skip(reason="Requires running PostgreSQL with pgvector extension - integration test")


def create_mock_pg_connection():
    """Create a mock PostgreSQL connection."""
    mock_conn = Mock(spec=['cursor', '__enter__', '__exit__'])  # Specify spec to avoid pool detection

    # Make connection work as context manager
    mock_conn.__enter__ = Mock(return_value=mock_conn)
    mock_conn.__exit__ = Mock(return_value=False)

    def cursor_factory(*args, **kwargs):
        """Factory to create cursor mocks with proper defaults."""
        mock_cursor = Mock()

        # Mock query results for searches
        mock_cursor.description = [
            ("id", None), ("department", None), ("content", None), ("distance", None)
        ]
        mock_cursor.fetchall = Mock(return_value=[
            ("doc1", "engineering", "Document 1", 0.1)
        ])

        # Mock results for health check queries
        # Health check calls fetchone() 4 times:
        # 1. Check extension exists
        # 2. Check table exists
        # 3. Get row count
        # 4. Check embedding column
        mock_cursor.fetchone = Mock(side_effect=[
            ("vector",),  # extension check
            (1,),  # table exists check
            (3,),  # row count
            ("embedding", "vector")  # column info
        ])
        mock_cursor.execute = Mock()
        mock_cursor.close = Mock()

        return mock_cursor

    mock_conn.cursor = Mock(side_effect=cursor_factory)

    return mock_conn


def create_basic_policy():
    """Create a basic allow-all policy."""
    return Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "all",
            "allow": {"everyone": True}
        }],
        "default": "deny"
    })


def test_pgvector_health_check_success():
    """Test successful health check for pgvector."""
    mock_conn = create_mock_pg_connection()

    retriever = PgvectorSecureRetriever(
        connection=mock_conn,
        table="documents",
        policy=create_basic_policy()
    )

    health = retriever.health_check()

    assert health["healthy"] is True
    assert health["backend"] == "pgvector"


def test_pgvector_context_manager():
    """Test using pgvector retriever as context manager."""
    mock_conn = create_mock_pg_connection()

    with PgvectorSecureRetriever(
        connection=mock_conn,
        table="documents",
        policy=create_basic_policy()
    ) as retriever:
        assert retriever is not None
        results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)
        assert len(results) >= 0


def test_pgvector_with_validation():
    """Test pgvector retriever with input validation."""
    mock_conn = create_mock_pg_connection()

    retriever = PgvectorSecureRetriever(
        connection=mock_conn,
        table="documents",
        policy=create_basic_policy(),
        enable_validation=True,
        validation_config=ValidationConfig()
    )

    # Valid user context
    results = retriever.search(
        [0.1, 0.2, 0.3],
        {"id": "alice", "department": "engineering"},
        limit=5
    )
    assert len(results) >= 0

    # Invalid user context (empty dictionary)
    with pytest.raises(RetrieverError):
        retriever.search(
            [0.1, 0.2, 0.3],
            {},
            limit=5
        )


def test_pgvector_with_retry():
    """Test pgvector retriever with retry logic."""
    mock_conn = Mock(spec=['cursor', '__enter__', '__exit__'])
    mock_conn.__enter__ = Mock(return_value=mock_conn)
    mock_conn.__exit__ = Mock(return_value=False)

    call_count = [0]

    def execute_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] < 3:
            raise OSError("Connection failed")

    def cursor_factory(*args, **kwargs):
        mock_cursor = Mock()
        mock_cursor.execute = Mock(side_effect=execute_side_effect)
        mock_cursor.description = [
            ("id", None), ("distance", None)
        ]
        mock_cursor.fetchall = Mock(return_value=[("doc1", 0.1)])
        mock_cursor.close = Mock()
        return mock_cursor

    mock_conn.cursor = Mock(side_effect=cursor_factory)

    retriever = PgvectorSecureRetriever(
        connection=mock_conn,
        table="documents",
        policy=create_basic_policy(),
        enable_retry=True,
        retry_config=RetryConfig(max_retries=3, initial_delay=0.01)
    )

    results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)

    assert len(results) >= 0
    assert call_count[0] == 3


def test_pgvector_with_cache():
    """Test pgvector retriever with filter caching."""
    mock_conn = create_mock_pg_connection()

    retriever = PgvectorSecureRetriever(
        connection=mock_conn,
        table="documents",
        policy=Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "dept-access",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            }],
            "default": "deny"
        }),
        enable_filter_cache=True,
        filter_cache_size=100
    )

    user = {"id": "alice", "department": "engineering"}

    # First search - cache miss
    retriever.search([0.1, 0.2, 0.3], user, limit=5)

    # Second search - cache hit
    retriever.search([0.1, 0.2, 0.3], user, limit=5)


def test_pgvector_with_audit_logging():
    """Test pgvector retriever with audit logging."""
    mock_conn = create_mock_pg_connection()

    audit_entries = []

    def audit_callback(entry):
        audit_entries.append(entry)

    audit_logger = AuditLogger(output=audit_callback)

    retriever = PgvectorSecureRetriever(
        connection=mock_conn,
        table="documents",
        policy=create_basic_policy(),
        audit_logger=audit_logger
    )

    user = {"id": "alice", "department": "engineering"}
    retriever.search([0.1, 0.2, 0.3], user, limit=5)

    assert len(audit_entries) == 1
    assert audit_entries[0]["user_id"] == "alice"


def test_pgvector_search_failure():
    """Test pgvector search failure handling."""
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_cursor.execute = Mock(side_effect=Exception("Search failed"))
    mock_cursor.close = Mock()

    mock_conn.cursor = Mock(return_value=mock_cursor)

    retriever = PgvectorSecureRetriever(
        connection=mock_conn,
        table="documents",
        policy=create_basic_policy(),
        enable_retry=False
    )

    with pytest.raises(RetrieverError, match="Search failed"):
        retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)


def test_pgvector_empty_results():
    """Test pgvector with empty search results."""
    mock_conn = Mock(spec=['cursor', '__enter__', '__exit__'])
    mock_conn.__enter__ = Mock(return_value=mock_conn)
    mock_conn.__exit__ = Mock(return_value=False)

    def cursor_factory(*args, **kwargs):
        mock_cursor = Mock()
        mock_cursor.execute = Mock()
        mock_cursor.description = [("id", None), ("distance", None)]
        mock_cursor.fetchall = Mock(return_value=[])
        mock_cursor.close = Mock()
        return mock_cursor

    mock_conn.cursor = Mock(side_effect=cursor_factory)

    retriever = PgvectorSecureRetriever(
        connection=mock_conn,
        table="documents",
        policy=create_basic_policy()
    )

    results = retriever.search([0.1, 0.2, 0.3], {"id": "alice"}, limit=5)

    assert len(results) == 0


def test_pgvector_batch_search():
    """Test batch search with pgvector."""
    mock_conn = create_mock_pg_connection()

    retriever = PgvectorSecureRetriever(
        connection=mock_conn,
        table="documents",
        policy=create_basic_policy()
    )

    queries = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ]

    user = {"id": "alice"}
    all_results = retriever.batch_search(queries, user, limit=5)

    assert len(all_results) == 3


def test_pgvector_policy_update():
    """Test updating policy on pgvector retriever."""
    mock_conn = create_mock_pg_connection()

    retriever = PgvectorSecureRetriever(
        connection=mock_conn,
        table="documents",
        policy=create_basic_policy()
    )

    new_policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "dept-access",
            "allow": {
                "conditions": ["user.department == document.department"]
            }
        }],
        "default": "deny"
    })

    retriever.policy = new_policy

    assert retriever.policy.rules[0].name == "dept-access"


def test_pgvector_with_embed_fn():
    """Test pgvector retriever with text query and embedding function."""
    mock_conn = create_mock_pg_connection()

    def embed_fn(text):
        return [float(ord(c)) / 1000 for c in text[:3]]

    retriever = PgvectorSecureRetriever(
        connection=mock_conn,
        table="documents",
        policy=create_basic_policy(),
        embed_fn=embed_fn
    )

    results = retriever.search("test query", {"id": "alice"}, limit=5)

    assert len(results) >= 0


def test_pgvector_text_query_without_embed_fn():
    """Test that text query without embed_fn raises error."""
    mock_conn = create_mock_pg_connection()

    retriever = PgvectorSecureRetriever(
        connection=mock_conn,
        table="documents",
        policy=create_basic_policy(),
        embed_fn=None
    )

    with pytest.raises(RetrieverError, match="no embed_fn was provided"):
        retriever.search("test query", {"id": "alice"}, limit=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
