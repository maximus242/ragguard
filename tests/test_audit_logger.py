"""
Tests for audit logging.
"""

import json
import tempfile
from pathlib import Path

import pytest

# Skip all tests if chromadb is not installed (required for TestFailOnAuditError tests)
pytest.importorskip("chromadb", exc_type=ImportError)

from ragguard.audit import AuditLogger, NullAuditLogger


def test_audit_logger_stdout(capsys):
    """Test audit logger with stdout output."""
    logger = AuditLogger(output="stdout")

    user = {"id": "test@example.com", "roles": ["user"]}
    logger.log(
        user=user,
        query="test query",
        results_count=5,
        filter_applied="some_filter"
    )

    captured = capsys.readouterr()
    output = json.loads(captured.out.strip())

    assert output["user_id"] == "test@example.com"
    assert output["user_roles"] == ["user"]
    assert output["query"] == "test query"
    assert output["results_returned"] == 5
    assert "timestamp" in output


def test_audit_logger_file():
    """Test audit logger with file output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "audit.jsonl"
        logger = AuditLogger(output=f"file:{log_file}")

        user = {"id": "test@example.com", "roles": ["user"]}
        logger.log(
            user=user,
            query="query 1",
            results_count=3,
            filter_applied="filter1"
        )
        logger.log(
            user=user,
            query="query 2",
            results_count=7,
            filter_applied="filter2"
        )

        # Read the log file
        with open(log_file) as f:
            lines = f.readlines()

        assert len(lines) == 2

        entry1 = json.loads(lines[0])
        assert entry1["query"] == "query 1"
        assert entry1["results_returned"] == 3

        entry2 = json.loads(lines[1])
        assert entry2["query"] == "query 2"
        assert entry2["results_returned"] == 7


def test_audit_logger_callback():
    """Test audit logger with callback."""
    logs = []

    def callback(entry):
        logs.append(entry)

    logger = AuditLogger(output=callback)

    user = {"id": "test@example.com", "roles": ["admin"]}
    logger.log(
        user=user,
        query="callback query",
        results_count=10,
        filter_applied="callback_filter"
    )

    assert len(logs) == 1
    assert logs[0]["query"] == "callback query"
    assert logs[0]["results_returned"] == 10


def test_audit_logger_additional_info():
    """Test audit logger with additional information."""
    logs = []

    def callback(entry):
        logs.append(entry)

    logger = AuditLogger(output=callback)

    user = {"id": "test@example.com"}
    logger.log(
        user=user,
        query="test",
        results_count=5,
        filter_applied="filter",
        additional_info={"custom_field": "custom_value", "latency_ms": 42}
    )

    assert len(logs) == 1
    assert logs[0]["custom_field"] == "custom_value"
    assert logs[0]["latency_ms"] == 42


def test_null_audit_logger():
    """Test that null logger does nothing."""
    logger = NullAuditLogger()

    # Should not raise any errors
    user = {"id": "test@example.com"}
    logger.log(
        user=user,
        query="test",
        results_count=5,
        filter_applied="filter"
    )

    # No way to verify it did nothing, but it shouldn't error


def test_audit_logger_creates_parent_directory():
    """Test that audit logger creates parent directories if needed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "subdir" / "audit.jsonl"
        logger = AuditLogger(output=f"file:{log_file}")

        user = {"id": "test@example.com"}
        logger.log(
            user=user,
            query="test",
            results_count=1,
            filter_applied="filter"
        )

        assert log_file.exists()
        assert log_file.parent.exists()


class TestFailOnAuditError:
    """Tests for fail_on_audit_error configuration."""

    def test_fail_open_by_default(self):
        """Test that audit failures are silently ignored by default (fail-open)."""
        from unittest.mock import Mock

        from ragguard import Policy
        from ragguard.retrievers import ChromaDBSecureRetriever

        # Create a failing audit logger
        class FailingLogger(AuditLogger):
            def log(self, **kwargs):
                raise RuntimeError("Audit log write failed")

        mock_collection = Mock()
        mock_collection.query = Mock(return_value={
            'ids': [['doc1']],
            'metadatas': [[{'visibility': 'public'}]],
            'documents': [['text']],
            'distances': [[0.1]]
        })

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "public", "allow": {"conditions": ["document.visibility == 'public'"]}}],
            "default": "deny"
        })

        retriever = ChromaDBSecureRetriever(
            collection=mock_collection,
            policy=policy,
            audit_logger=FailingLogger(output="stdout"),
            # fail_on_audit_error=False is the default
        )

        # Should succeed despite audit failure (fail-open)
        results = retriever.search(
            query=[0.1, 0.2, 0.3],
            user={"id": "alice"},
            limit=10
        )

        assert len(results) == 1

    def test_fail_closed_mode_raises_on_audit_error(self):
        """Test that fail_on_audit_error=True raises AuditLogError on failure."""
        from unittest.mock import Mock

        from ragguard import AuditLogError, Policy
        from ragguard.retrievers import ChromaDBSecureRetriever

        # Create a failing audit logger
        class FailingLogger(AuditLogger):
            def log(self, **kwargs):
                raise RuntimeError("Audit log write failed")

        mock_collection = Mock()
        mock_collection.query = Mock(return_value={
            'ids': [['doc1']],
            'metadatas': [[{'visibility': 'public'}]],
            'documents': [['text']],
            'distances': [[0.1]]
        })

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "public", "allow": {"conditions": ["document.visibility == 'public'"]}}],
            "default": "deny"
        })

        retriever = ChromaDBSecureRetriever(
            collection=mock_collection,
            policy=policy,
            audit_logger=FailingLogger(output="stdout"),
            fail_on_audit_error=True  # Enable fail-closed mode
        )

        # Should raise AuditLogError when audit fails
        with pytest.raises(AuditLogError) as exc_info:
            retriever.search(
                query=[0.1, 0.2, 0.3],
                user={"id": "alice"},
                limit=10
            )

        assert "Audit logging failed" in str(exc_info.value)
        assert exc_info.value.cause is not None

    def test_fail_closed_with_search_error(self):
        """Test that fail_on_audit_error works for search error audit logging too."""
        from unittest.mock import Mock

        from ragguard import AuditLogError, Policy
        from ragguard.retrievers import ChromaDBSecureRetriever

        # Create a logger that fails
        class FailingLogger(AuditLogger):
            def log(self, **kwargs):
                raise RuntimeError("Audit log write failed")

        mock_collection = Mock()
        mock_collection.query = Mock(side_effect=Exception("Database error"))

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "public", "allow": {"conditions": ["document.visibility == 'public'"]}}],
            "default": "deny"
        })

        retriever = ChromaDBSecureRetriever(
            collection=mock_collection,
            policy=policy,
            audit_logger=FailingLogger(output="stdout"),
            fail_on_audit_error=True
        )

        # Should raise AuditLogError (not the original database error)
        with pytest.raises(AuditLogError) as exc_info:
            retriever.search(
                query=[0.1, 0.2, 0.3],
                user={"id": "alice"},
                limit=10
            )

        # The audit error should be raised, indicating audit failure took priority
        assert "Audit logging failed" in str(exc_info.value)

    def test_successful_audit_works_normally(self):
        """Test that normal audit logging still works with fail_on_audit_error=True."""
        from unittest.mock import Mock

        from ragguard import Policy
        from ragguard.retrievers import ChromaDBSecureRetriever

        logs = []

        class WorkingLogger(AuditLogger):
            def log(self, **kwargs):
                logs.append(kwargs)

        mock_collection = Mock()
        mock_collection.query = Mock(return_value={
            'ids': [['doc1']],
            'metadatas': [[{'visibility': 'public'}]],
            'documents': [['text']],
            'distances': [[0.1]]
        })

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{"name": "public", "allow": {"conditions": ["document.visibility == 'public'"]}}],
            "default": "deny"
        })

        retriever = ChromaDBSecureRetriever(
            collection=mock_collection,
            policy=policy,
            audit_logger=WorkingLogger(output="stdout"),
            fail_on_audit_error=True
        )

        # Should succeed with working logger
        results = retriever.search(
            query=[0.1, 0.2, 0.3],
            user={"id": "alice"},
            limit=10
        )

        assert len(results) == 1
        assert len(logs) == 1
        assert logs[0]["results_count"] == 1
