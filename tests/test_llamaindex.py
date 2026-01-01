"""
Tests for LlamaIndex integration.
"""

import sys
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from ragguard.policy.models import AllowConditions, Policy, Rule

# Check if llama-index is installed
try:
    import llama_index
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False


@pytest.fixture
def sample_policy():
    """Create a sample policy for testing."""
    return Policy(
        version="1",
        default="deny",
        rules=[
            Rule(
                name="dept-access",
                allow=AllowConditions(
                    conditions=["user.department == document.department"]
                ),
            ),
            Rule(
                name="public-docs",
                match={"visibility": "public"},
                allow=AllowConditions(everyone=True),
            ),
        ],
    )


class TestLlamaIndexImport:
    """Test LlamaIndex module import."""

    def test_import_without_llamaindex(self):
        """Test that import works even without llama_index installed."""
        # This should not raise
        from ragguard.integrations import llamaindex

        # Check for the actual class names
        assert hasattr(llamaindex, "SecureLlamaIndexRetriever")
        assert hasattr(llamaindex, "SecureQueryEngine")
        assert hasattr(llamaindex, "wrap_retriever")


class TestSecureLlamaIndexRetriever:
    """Test LlamaIndex secure retriever."""

    @pytest.mark.skipif(LLAMAINDEX_AVAILABLE, reason="Test only runs when llama-index is NOT installed")
    def test_initialization_requires_llamaindex(self, sample_policy):
        """Test that initialization fails without llama-index installed."""
        from ragguard.exceptions import RetrieverError
        from ragguard.integrations.llamaindex import SecureLlamaIndexRetriever

        mock_retriever = MagicMock()

        # Should raise because llama-index is not installed
        with pytest.raises(RetrieverError, match="llama-index not installed"):
            SecureLlamaIndexRetriever(
                base_retriever=mock_retriever,
                policy=sample_policy,
            )



class TestSecureQueryEngine:
    """Test LlamaIndex secure query engine."""

    @pytest.mark.skipif(LLAMAINDEX_AVAILABLE, reason="Test only runs when llama-index is NOT installed")
    def test_initialization_requires_llamaindex(self, sample_policy):
        """Test that initialization fails without llama-index installed."""
        from ragguard.exceptions import RetrieverError
        from ragguard.integrations.llamaindex import SecureQueryEngine

        mock_index = MagicMock()

        # Should raise because llama-index is not installed
        with pytest.raises(RetrieverError, match="llama-index not installed"):
            SecureQueryEngine(
                index=mock_index,
                policy=sample_policy,
            )


class TestWrapRetriever:
    """Test the wrap_retriever utility function."""

    @pytest.mark.skipif(LLAMAINDEX_AVAILABLE, reason="Test only runs when llama-index is NOT installed")
    def test_wrap_retriever_requires_llamaindex(self, sample_policy):
        """Test that wrap_retriever fails without llama-index installed."""
        from ragguard.exceptions import RetrieverError
        from ragguard.integrations.llamaindex import wrap_retriever

        mock_retriever = MagicMock()

        with pytest.raises(RetrieverError, match="llama-index not installed"):
            wrap_retriever(
                retriever=mock_retriever,
                policy=sample_policy,
            )


class TestPolicyFiltering:
    """Test policy filtering in LlamaIndex integration."""

    def test_filtering_concepts(self, sample_policy):
        """Test that the policy filtering concepts are correct."""
        from ragguard.policy.engine import PolicyEngine

        engine = PolicyEngine(sample_policy)

        # Test department match
        user = {"id": "alice", "department": "engineering"}
        doc_eng = {"department": "engineering"}
        doc_marketing = {"department": "marketing"}

        assert engine.evaluate(user, doc_eng) is True
        assert engine.evaluate(user, doc_marketing) is False

    def test_public_docs_accessible(self, sample_policy):
        """Test that public docs are accessible to everyone."""
        from ragguard.policy.engine import PolicyEngine

        engine = PolicyEngine(sample_policy)

        # Any user should be able to access public docs
        user = {"id": "bob", "department": "sales"}
        doc_public = {"visibility": "public", "department": "engineering"}

        assert engine.evaluate(user, doc_public) is True

    def test_filtered_results_format(self, sample_policy):
        """Test the expected format of filtered results."""
        from ragguard.policy.engine import PolicyEngine

        engine = PolicyEngine(sample_policy)

        # Simulate what the retriever would do
        user = {"id": "alice", "department": "engineering"}

        mock_results = [
            {"node": {"metadata": {"department": "engineering"}}, "score": 0.9},
            {"node": {"metadata": {"department": "marketing"}}, "score": 0.8},
            {"node": {"metadata": {"visibility": "public"}}, "score": 0.7},
        ]

        filtered = [
            r
            for r in mock_results
            if engine.evaluate(user, r["node"]["metadata"])
        ]

        # Should get engineering doc and public doc
        assert len(filtered) == 2
        assert filtered[0]["node"]["metadata"]["department"] == "engineering"
        assert filtered[1]["node"]["metadata"].get("visibility") == "public"
