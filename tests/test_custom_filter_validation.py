"""
Tests for custom filter builder validation.

These tests ensure that custom filter builders are validated
for common issues that could lead to security problems.
"""

import logging
from unittest.mock import Mock, patch

import pytest

# Skip all tests if chromadb is not installed
pytest.importorskip("chromadb", exc_type=ImportError)

from ragguard.policy import Policy
from ragguard.retrievers import ChromaDBSecureRetriever


@pytest.fixture
def sample_policy():
    """Create a sample policy for testing."""
    return Policy.from_dict({
        "version": "1",
        "rules": [{"name": "allow-all", "allow": {"everyone": True}}],
        "default": "deny"
    })


@pytest.fixture
def mock_collection():
    """Create a mock ChromaDB collection."""
    mock = Mock()
    mock.query = Mock(return_value={
        "ids": [["doc1"]],
        "distances": [[0.1]],
        "metadatas": [[{"visibility": "public"}]],
        "documents": [["text"]]
    })
    return mock


class TestCustomFilterBuilderValidation:
    """Test validation of custom filter builder results."""

    def test_empty_dict_logs_warning(self, sample_policy, mock_collection):
        """Test that empty dict filter logs a warning."""
        # Create custom builder that returns empty dict
        custom_builder = Mock()
        custom_builder.build_filter = Mock(return_value={})

        retriever = ChromaDBSecureRetriever(
            collection=mock_collection,
            policy=sample_policy,
            custom_filter_builder=custom_builder
        )

        user = {"id": "alice"}

        # Patch the logger to capture the warning
        with patch("ragguard.retrievers.base.logger") as mock_logger:
            retriever.preview_filter(user)
            # Check that warning was called with expected message
            mock_logger.warning.assert_called()
            call_args = str(mock_logger.warning.call_args)
            assert "empty dict" in call_args.lower()

    def test_empty_list_logs_warning(self, sample_policy, mock_collection):
        """Test that empty list filter logs a warning (either empty list or wrong type)."""
        custom_builder = Mock()
        custom_builder.build_filter = Mock(return_value=[])

        retriever = ChromaDBSecureRetriever(
            collection=mock_collection,
            policy=sample_policy,
            custom_filter_builder=custom_builder
        )

        user = {"id": "alice"}

        with patch("ragguard.retrievers.base.logger") as mock_logger:
            retriever.preview_filter(user)
            mock_logger.warning.assert_called()
            call_args = str(mock_logger.warning.call_args)
            # Either "empty list" or "wrong type" warning is acceptable
            # since list is wrong type for chromadb (expects dict)
            assert "empty list" in call_args.lower() or "wrong type" in call_args.lower()

    def test_wrong_type_for_chromadb_logs_warning(self, sample_policy, mock_collection):
        """Test that wrong type for chromadb logs a warning."""
        # ChromaDB expects dict, not string
        custom_builder = Mock()
        custom_builder.build_filter = Mock(return_value="WHERE status = 'active'")

        retriever = ChromaDBSecureRetriever(
            collection=mock_collection,
            policy=sample_policy,
            custom_filter_builder=custom_builder
        )

        user = {"id": "alice"}

        with patch("ragguard.retrievers.base.logger") as mock_logger:
            retriever.preview_filter(user)
            mock_logger.warning.assert_called()
            call_args = str(mock_logger.warning.call_args)
            assert "wrong type" in call_args.lower()

    def test_valid_dict_no_warning(self, sample_policy, mock_collection):
        """Test that valid dict filter does not log warning."""
        custom_builder = Mock()
        custom_builder.build_filter = Mock(return_value={"status": {"$eq": "active"}})

        retriever = ChromaDBSecureRetriever(
            collection=mock_collection,
            policy=sample_policy,
            custom_filter_builder=custom_builder
        )

        user = {"id": "alice"}

        with patch("ragguard.retrievers.base.logger") as mock_logger:
            retriever.preview_filter(user)
            # Should not have called warning for filter issues
            # (may have other debug calls, but not for filter type issues)
            for call in mock_logger.warning.call_args_list:
                call_str = str(call).lower()
                assert "wrong type" not in call_str
                assert "empty dict" not in call_str

    def test_none_filter_no_warning(self, sample_policy, mock_collection):
        """Test that None filter does not log warning (None is valid for allow-all)."""
        custom_builder = Mock()
        custom_builder.build_filter = Mock(return_value=None)

        retriever = ChromaDBSecureRetriever(
            collection=mock_collection,
            policy=sample_policy,
            custom_filter_builder=custom_builder
        )

        user = {"id": "alice"}

        with patch("ragguard.retrievers.base.logger") as mock_logger:
            retriever.preview_filter(user)
            # Should not have called warning for filter type
            for call in mock_logger.warning.call_args_list:
                call_str = str(call).lower()
                assert "wrong type" not in call_str

    def test_validation_includes_builder_name(self, sample_policy, mock_collection):
        """Test that validation logs include the builder class name."""

        class MyCustomBuilder:
            def build_filter(self, policy, user, backend):
                return {}  # Empty dict (triggers warning)

        retriever = ChromaDBSecureRetriever(
            collection=mock_collection,
            policy=sample_policy,
            custom_filter_builder=MyCustomBuilder()
        )

        user = {"id": "alice"}

        with patch("ragguard.retrievers.base.logger") as mock_logger:
            retriever.preview_filter(user)
            mock_logger.warning.assert_called()
            # Check that extra_fields contains builder name
            call_args = mock_logger.warning.call_args
            if call_args.kwargs.get("extra"):
                extra_fields = call_args.kwargs["extra"].get("extra_fields", {})
                assert extra_fields.get("builder") == "MyCustomBuilder"


class TestCustomFilterBuilderValidationInSearch:
    """Test that validation also runs during search."""

    def test_search_validates_custom_filter(self, sample_policy, mock_collection):
        """Test that search also validates custom filter results."""
        custom_builder = Mock()
        custom_builder.build_filter = Mock(return_value={})  # Empty dict

        retriever = ChromaDBSecureRetriever(
            collection=mock_collection,
            policy=sample_policy,
            custom_filter_builder=custom_builder
        )

        user = {"id": "alice"}
        query_vector = [0.1] * 384  # Dummy vector

        with patch("ragguard.retrievers.base.logger") as mock_logger:
            retriever.search(query_vector, user, limit=10)
            # Validation warning should appear
            mock_logger.warning.assert_called()
            call_args = str(mock_logger.warning.call_args)
            assert "empty dict" in call_args.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
