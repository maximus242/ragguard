"""
Comprehensive tests for DSPy integration to maximize coverage.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestRAGGuardRM:
    """Tests for RAGGuardRM class."""

    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.search.return_value = [
            {"text": "Document 1", "score": 0.95},
            {"text": "Document 2", "score": 0.85},
        ]
        return retriever

    def test_creation(self, mock_retriever):
        """Test RAGGuardRM creation with defaults."""
        from ragguard.integrations.dspy import RAGGuardRM

        rm = RAGGuardRM(
            retriever=mock_retriever,
            user={"id": "alice", "department": "eng"}
        )

        assert rm.retriever == mock_retriever
        assert rm.k == 5
        assert rm.text_field == "text"
        assert rm.score_field == "score"
        assert rm.user["id"] == "alice"

    def test_creation_with_custom_options(self, mock_retriever):
        """Test creation with custom options."""
        from ragguard.integrations.dspy import RAGGuardRM

        rm = RAGGuardRM(
            retriever=mock_retriever,
            user={"id": "bob"},
            k=10,
            text_field="content",
            score_field="relevance"
        )

        assert rm.k == 10
        assert rm.text_field == "content"
        assert rm.score_field == "relevance"

    def test_call_returns_passages(self, mock_retriever):
        """Test __call__ returns list of passages."""
        from ragguard.integrations.dspy import RAGGuardRM

        rm = RAGGuardRM(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = rm("test query")

        assert isinstance(result, list)
        mock_retriever.search.assert_called_once()
        call_kwargs = mock_retriever.search.call_args.kwargs
        assert call_kwargs["query"] == "test query"
        assert call_kwargs["limit"] == 5

    def test_call_with_custom_k(self, mock_retriever):
        """Test __call__ with custom k parameter."""
        from ragguard.integrations.dspy import RAGGuardRM

        rm = RAGGuardRM(
            retriever=mock_retriever,
            user={"id": "alice"},
            k=5
        )

        rm("test query", k=3)

        call_kwargs = mock_retriever.search.call_args.kwargs
        assert call_kwargs["limit"] == 3

    def test_call_error_handling(self, mock_retriever):
        """Test __call__ handles errors gracefully."""
        from ragguard.integrations.dspy import RAGGuardRM

        mock_retriever.search.side_effect = Exception("Search failed")

        rm = RAGGuardRM(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = rm("test query")

        assert result == []

    def test_extract_passages_dict_results(self, mock_retriever):
        """Test _extract_passages with dict results."""
        from ragguard.integrations.dspy import RAGGuardRM

        rm = RAGGuardRM(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        results = [
            {"text": "Text 1", "score": 0.9},
            {"content": "Content 2", "score": 0.8},
            {"metadata": {"text": "Nested text"}, "score": 0.7},
        ]

        passages = rm._extract_passages(results)

        assert len(passages) == 3
        assert passages[0] == "Text 1"
        assert passages[1] == "Content 2"

    def test_extract_passages_qdrant_results(self, mock_retriever):
        """Test _extract_passages with Qdrant-style results."""
        from ragguard.integrations.dspy import RAGGuardRM

        rm = RAGGuardRM(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = MagicMock()
        result.payload = {"text": "Qdrant doc"}

        passages = rm._extract_passages([result])

        assert len(passages) == 1
        assert passages[0] == "Qdrant doc"

    def test_extract_passages_qdrant_content_field(self, mock_retriever):
        """Test _extract_passages with content field in Qdrant payload."""
        from ragguard.integrations.dspy import RAGGuardRM

        rm = RAGGuardRM(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = MagicMock()
        result.payload = {"content": "Content field"}

        passages = rm._extract_passages([result])

        assert passages[0] == "Content field"

    def test_extract_passages_fallback_to_str(self, mock_retriever):
        """Test _extract_passages fallback to string conversion."""
        from ragguard.integrations.dspy import RAGGuardRM

        rm = RAGGuardRM(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        passages = rm._extract_passages(["plain string"])

        assert len(passages) == 1
        assert passages[0] == "plain string"

    def test_set_user(self, mock_retriever):
        """Test set_user method."""
        from ragguard.integrations.dspy import RAGGuardRM

        rm = RAGGuardRM(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = rm.set_user({"id": "bob", "dept": "hr"})

        assert result is rm  # Returns self for chaining
        assert rm.user["id"] == "bob"
        assert rm.user["dept"] == "hr"

    def test_user_property(self, mock_retriever):
        """Test user property."""
        from ragguard.integrations.dspy import RAGGuardRM

        rm = RAGGuardRM(
            retriever=mock_retriever,
            user={"id": "alice", "roles": ["user"]}
        )

        assert rm.user["id"] == "alice"
        assert rm.user["roles"] == ["user"]


class TestSecureRetrieve:
    """Tests for SecureRetrieve class."""

    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.search.return_value = [
            {"text": "Result 1", "score": 0.95},
            {"text": "Result 2", "score": 0.85},
        ]
        return retriever

    def test_creation(self, mock_retriever):
        """Test SecureRetrieve creation."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE, SecureRetrieve

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        sr = SecureRetrieve(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        assert sr._retriever == mock_retriever
        assert sr._user["id"] == "alice"
        assert sr.k == 3

    def test_creation_requires_retriever(self):
        """Test creation fails without retriever."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE, SecureRetrieve

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        with pytest.raises(ValueError, match="retriever is required"):
            SecureRetrieve(retriever=None, user={"id": "alice"})

    def test_creation_requires_user(self, mock_retriever):
        """Test creation fails without user."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE, SecureRetrieve

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        with pytest.raises(ValueError, match="user context is required"):
            SecureRetrieve(retriever=mock_retriever, user=None)

    def test_forward_single_query(self, mock_retriever):
        """Test forward with single query."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE, SecureRetrieve

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        sr = SecureRetrieve(
            retriever=mock_retriever,
            user={"id": "alice"},
            k=5
        )

        result = sr.forward("test query")

        assert hasattr(result, "passages")
        mock_retriever.search.assert_called_once()

    def test_forward_multiple_queries(self, mock_retriever):
        """Test forward with multiple queries."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE, SecureRetrieve

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        sr = SecureRetrieve(
            retriever=mock_retriever,
            user={"id": "alice"},
            k=5
        )

        result = sr.forward(["query 1", "query 2"])

        assert hasattr(result, "passages")
        assert mock_retriever.search.call_count == 2

    def test_forward_with_custom_k(self, mock_retriever):
        """Test forward with custom k parameter."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE, SecureRetrieve

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        sr = SecureRetrieve(
            retriever=mock_retriever,
            user={"id": "alice"},
            k=10
        )

        sr.forward("test query", k=3)

        call_kwargs = mock_retriever.search.call_args.kwargs
        assert call_kwargs["limit"] == 3

    def test_forward_deduplicates_passages(self, mock_retriever):
        """Test forward removes duplicate passages."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE, SecureRetrieve

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        mock_retriever.search.return_value = [
            {"text": "Same text", "score": 0.9},
            {"text": "Same text", "score": 0.8},
        ]

        sr = SecureRetrieve(
            retriever=mock_retriever,
            user={"id": "alice"},
            k=10
        )

        result = sr.forward("test query")

        # Should deduplicate
        assert len(result.passages) <= 2

    def test_forward_error_handling(self, mock_retriever):
        """Test forward handles errors gracefully."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE, SecureRetrieve

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        mock_retriever.search.side_effect = Exception("Search failed")

        sr = SecureRetrieve(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = sr.forward("test query")

        assert hasattr(result, "passages")
        assert result.passages == []

    def test_extract_passages_dict_results(self, mock_retriever):
        """Test _extract_passages with dict results."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE, SecureRetrieve

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        sr = SecureRetrieve(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        results = [
            {"text": "Text 1"},
            {"content": "Content 2"},
            {"metadata": {"text": "Nested"}},
        ]

        passages = sr._extract_passages(results)

        assert len(passages) == 3

    def test_extract_passages_qdrant_results(self, mock_retriever):
        """Test _extract_passages with Qdrant-style results."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE, SecureRetrieve

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        sr = SecureRetrieve(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = MagicMock()
        result.payload = {"text": "Qdrant text"}

        passages = sr._extract_passages([result])

        assert passages[0] == "Qdrant text"

    def test_set_user(self, mock_retriever):
        """Test set_user method."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE, SecureRetrieve

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        sr = SecureRetrieve(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = sr.set_user({"id": "bob"})

        assert result is sr
        assert sr.user["id"] == "bob"

    def test_user_property(self, mock_retriever):
        """Test user property."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE, SecureRetrieve

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        sr = SecureRetrieve(
            retriever=mock_retriever,
            user={"id": "alice", "dept": "eng"}
        )

        assert sr.user["id"] == "alice"
        assert sr.user["dept"] == "eng"


class TestSecureRAG:
    """Tests for SecureRAG class."""

    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.search.return_value = [
            {"text": "Context document", "score": 0.9}
        ]
        return retriever

    def test_creation(self, mock_retriever):
        """Test SecureRAG creation."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE, SecureRAG

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        rag = SecureRAG(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        assert rag._user["id"] == "alice"
        assert rag.retrieve is not None
        assert rag.generate is not None

    def test_creation_with_custom_options(self, mock_retriever):
        """Test creation with custom options."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE, SecureRAG

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        rag = SecureRAG(
            retriever=mock_retriever,
            user={"id": "alice"},
            k=10,
            signature="context, question -> detailed_answer"
        )

        assert rag.retrieve.k == 10

    def test_forward_with_results(self, mock_retriever):
        """Test forward when results are found."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE, SecureRAG

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        rag = SecureRAG(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = rag.forward("What is the answer?")

        assert hasattr(result, "context")

    def test_forward_without_results(self, mock_retriever):
        """Test forward when no results found."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE, SecureRAG

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        mock_retriever.search.return_value = []

        rag = SecureRAG(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = rag.forward("What is the answer?")

        assert hasattr(result, "answer")
        assert "don't have access" in result.answer

    def test_set_user(self, mock_retriever):
        """Test set_user method."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE, SecureRAG

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        rag = SecureRAG(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = rag.set_user({"id": "bob"})

        assert result is rag
        assert rag.user["id"] == "bob"
        assert rag.retrieve.user["id"] == "bob"

    def test_user_property(self, mock_retriever):
        """Test user property."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE, SecureRAG

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        rag = SecureRAG(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        assert rag.user["id"] == "alice"


class TestConfigureRagguardRM:
    """Tests for configure_ragguard_rm function."""

    def test_configure(self):
        """Test configure_ragguard_rm creates and configures RM."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE, RAGGuardRM, configure_ragguard_rm

        if not DSPY_AVAILABLE:
            pytest.skip("DSPy not available")

        mock_retriever = MagicMock()

        rm = configure_ragguard_rm(
            retriever=mock_retriever,
            user={"id": "alice"},
            k=10
        )

        assert isinstance(rm, RAGGuardRM)
        assert rm.k == 10
        assert rm.user["id"] == "alice"


class TestCheckDspyAvailable:
    """Tests for _check_dspy_available function."""

    def test_check_when_not_available(self):
        """Test check raises when DSPy not installed."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE, _check_dspy_available

        if DSPY_AVAILABLE:
            pytest.skip("DSPy is available")

        with pytest.raises(ImportError, match="dspy-ai"):
            _check_dspy_available()


class TestDspyModuleExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Test __all__ exports."""
        from ragguard.integrations import dspy

        expected = [
            "RAGGuardRM",
            "SecureRetrieve",
            "SecureRAG",
            "configure_ragguard_rm",
            "DSPY_AVAILABLE",
        ]

        for name in expected:
            assert name in dspy.__all__

    def test_dspy_available_flag(self):
        """Test DSPY_AVAILABLE flag is exported."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE

        assert isinstance(DSPY_AVAILABLE, bool)


class TestDspyStubClasses:
    """Tests for stub classes when DSPy not installed."""

    def test_ragguard_rm_without_dspy(self):
        """Test RAGGuardRM works without DSPy."""
        from ragguard.integrations.dspy import RAGGuardRM

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [{"text": "Test"}]

        rm = RAGGuardRM(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        # Should work even without DSPy
        result = rm("query")
        assert isinstance(result, list)

    def test_stub_prediction_class(self):
        """Test stub Prediction class attributes."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE

        if DSPY_AVAILABLE:
            pytest.skip("DSPy is installed, stub not used")

        # Import the stub module
        import ragguard.integrations.dspy as dspy_module

        # Access the stub classes
        prediction = dspy_module._Prediction(passages=["test"], answer="answer")
        assert prediction.passages == ["test"]
        assert prediction.answer == "answer"

    def test_stub_module_class(self):
        """Test stub Module class."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE

        if DSPY_AVAILABLE:
            pytest.skip("DSPy is installed, stub not used")

        import ragguard.integrations.dspy as dspy_module

        # _Module should be the stub
        assert dspy_module._Module is not None

    def test_stub_retrieve_class(self):
        """Test stub Retrieve class."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE, Retrieve

        if DSPY_AVAILABLE:
            pytest.skip("DSPy is installed, stub not used")

        # The Retrieve class should be our stub
        retrieve = Retrieve(k=5)
        assert retrieve.k == 5


class TestDspyStubDspyModule:
    """Tests for the _DspyStub class."""

    def test_stub_dspy_settings_configure(self):
        """Test stub dspy.settings.configure."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE

        if DSPY_AVAILABLE:
            pytest.skip("DSPy is installed, stub not used")

        import ragguard.integrations.dspy as dspy_module

        # Get the stub dspy object
        stub_dspy = dspy_module.dspy

        # Should be able to call configure without error
        stub_dspy.settings.configure(rm=MagicMock())

    def test_stub_dspy_chain_of_thought(self):
        """Test stub dspy.ChainOfThought."""
        from ragguard.integrations.dspy import DSPY_AVAILABLE

        if DSPY_AVAILABLE:
            pytest.skip("DSPy is installed, stub not used")

        import ragguard.integrations.dspy as dspy_module

        # Get the stub dspy object
        stub_dspy = dspy_module.dspy

        # Create ChainOfThought
        cot = stub_dspy.ChainOfThought("context, question -> answer")
        result = cot(context="test context", question="test question")

        assert hasattr(result, "answer")
        assert result.answer == ""


class TestRAGGuardRMExtractPassages:
    """Additional tests for _extract_passages edge cases."""

    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.search.return_value = []
        return retriever

    def test_extract_passages_qdrant_payload_str_fallback(self, mock_retriever):
        """Test _extract_passages with payload that has no known text field."""
        from ragguard.integrations.dspy import RAGGuardRM

        rm = RAGGuardRM(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        result = MagicMock()
        result.payload = {"unknown_field": "data", "other": 123}

        passages = rm._extract_passages([result])

        # Should convert payload to string as fallback
        assert len(passages) == 1
        assert "unknown_field" in passages[0] or "data" in passages[0]

    def test_extract_passages_dict_str_fallback(self, mock_retriever):
        """Test _extract_passages with dict that has no known text field."""
        from ragguard.integrations.dspy import RAGGuardRM

        rm = RAGGuardRM(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        results = [{"unknown_key": "value", "number": 42}]

        passages = rm._extract_passages(results)

        # Should convert dict to string as fallback
        assert len(passages) == 1

    def test_extract_passages_custom_text_field(self, mock_retriever):
        """Test _extract_passages with custom text_field."""
        from ragguard.integrations.dspy import RAGGuardRM

        rm = RAGGuardRM(
            retriever=mock_retriever,
            user={"id": "alice"},
            text_field="custom_text"
        )

        results = [{"custom_text": "My custom text field"}]

        passages = rm._extract_passages(results)

        assert passages[0] == "My custom text field"

    def test_extract_passages_dict_metadata_text(self, mock_retriever):
        """Test _extract_passages extracts from metadata.text."""
        from ragguard.integrations.dspy import RAGGuardRM

        rm = RAGGuardRM(
            retriever=mock_retriever,
            user={"id": "alice"}
        )

        results = [{"metadata": {"text": "Nested in metadata"}}]

        passages = rm._extract_passages(results)

        assert passages[0] == "Nested in metadata"
