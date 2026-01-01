"""
DSPy integration for RAGGuard.

Provides secure retriever modules for DSPy pipelines with permission-aware
document retrieval.

DSPy is Stanford NLP's framework for programming—rather than prompting—language
models. This integration allows DSPy programs to use RAGGuard retrievers
with automatic permission filtering.

Example:
    ```python
    import dspy
    from ragguard.integrations.dspy import RAGGuardRM, SecureRetrieve
    from ragguard import QdrantSecureRetriever, Policy

    # Create RAGGuard retriever
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "dept", "allow": {"conditions": ["user.department == document.department"]}}],
        "default": "deny"
    })

    retriever = QdrantSecureRetriever(
        client=qdrant_client,
        collection="documents",
        policy=policy,
        embed_fn=embed_function
    )

    # Create DSPy retriever model
    rm = RAGGuardRM(
        retriever=retriever,
        user={"id": "alice", "department": "engineering"},
        k=5
    )

    # Configure DSPy to use this retriever
    dspy.settings.configure(rm=rm)

    # Use in DSPy program
    class RAG(dspy.Module):
        def __init__(self):
            self.retrieve = dspy.Retrieve(k=3)
            self.generate = dspy.ChainOfThought("context, question -> answer")

        def forward(self, question):
            context = self.retrieve(question).passages
            return self.generate(context=context, question=question)

    rag = RAG()
    result = rag("What are the Q3 revenue numbers?")
    ```

Requirements:
    - dspy-ai>=2.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Check if DSPy is available
try:
    import dspy
    from dspy import Retrieve
    DSPY_AVAILABLE = True
    _Prediction = dspy.Prediction
    _Module = dspy.Module
except ImportError:
    DSPY_AVAILABLE = False

    # Stub classes when DSPy not installed
    class _DspyStub:
        """Stub for dspy module."""
        class Prediction:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        class Module:
            pass

        class settings:
            @staticmethod
            def configure(**kwargs):
                pass

        @staticmethod
        def ChainOfThought(signature):
            return lambda **kwargs: _DspyStub.Prediction(answer="")

    dspy = _DspyStub  # type: ignore
    _Prediction = _DspyStub.Prediction
    _Module = _DspyStub.Module

    class Retrieve:  # type: ignore
        """Stub class when DSPy not installed."""
        def __init__(self, k: int = 3):
            self.k = k


def _check_dspy_available():
    """Check if DSPy is installed."""
    if not DSPY_AVAILABLE:
        raise ImportError(
            "DSPy integration requires dspy-ai. "
            "Install with: pip install dspy-ai"
        )


class RAGGuardRM:
    """
    RAGGuard Retriever Model for DSPy.

    This class implements DSPy's retriever model interface, allowing
    RAGGuard secure retrievers to be used as the default retrieval
    backend in DSPy programs.

    Attributes:
        retriever: RAGGuard secure retriever instance
        user: User context for permission filtering
        k: Default number of results to retrieve

    Example:
        ```python
        # Create retriever model
        rm = RAGGuardRM(
            retriever=qdrant_retriever,
            user={"id": "alice", "department": "engineering"},
            k=5
        )

        # Configure DSPy
        dspy.settings.configure(rm=rm)

        # Now dspy.Retrieve() will use RAGGuard
        retrieve = dspy.Retrieve(k=3)
        results = retrieve("machine learning papers")
        ```
    """

    def __init__(
        self,
        retriever: Any,
        user: Dict[str, Any],
        k: int = 5,
        text_field: str = "text",
        score_field: str = "score"
    ):
        """
        Initialize RAGGuard retriever model.

        Args:
            retriever: RAGGuard secure retriever instance
            user: User context for permission filtering
            k: Default number of results
            text_field: Field name for document text in results
            score_field: Field name for relevance score
        """
        self.retriever = retriever
        self._user = user
        self.k = k
        self.text_field = text_field
        self.score_field = score_field

    def __call__(
        self,
        query: str,
        k: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """
        Retrieve documents for a query.

        This method is called by DSPy's Retrieve module.

        Args:
            query: Search query string
            k: Number of results (uses default if not provided)
            **kwargs: Additional search parameters

        Returns:
            List of document text strings
        """
        num_results = k or self.k

        try:
            results = self.retriever.search(
                query=query,
                user=self._user,
                limit=num_results,
                **kwargs
            )

            return self._extract_passages(results)

        except Exception as e:
            logger.error(f"RAGGuard retrieval failed: {e}")
            return []

    def _extract_passages(self, results: List[Any]) -> List[str]:
        """Extract text passages from retriever results."""
        passages = []

        for result in results:
            if hasattr(result, 'payload'):
                # Qdrant-style result
                text = (
                    result.payload.get(self.text_field) or
                    result.payload.get('content') or
                    result.payload.get('text') or
                    str(result.payload)
                )
            elif isinstance(result, dict):
                text = (
                    result.get(self.text_field) or
                    result.get('content') or
                    result.get('text') or
                    result.get('metadata', {}).get('text') or
                    str(result)
                )
            else:
                text = str(result)

            passages.append(text)

        return passages

    def set_user(self, user: Dict[str, Any]) -> "RAGGuardRM":
        """
        Update the user context.

        Args:
            user: New user context for permission filtering

        Returns:
            Self for method chaining
        """
        self._user = user
        return self

    @property
    def user(self) -> Dict[str, Any]:
        """Get current user context."""
        return self._user


class SecureRetrieve(Retrieve):
    """
    Secure retrieval module for DSPy with built-in permission filtering.

    This module extends DSPy's Retrieve to use a RAGGuard retriever
    directly, without needing to configure the global retriever model.

    Example:
        ```python
        import dspy
        from ragguard.integrations.dspy import SecureRetrieve

        class SecureRAG(dspy.Module):
            def __init__(self, retriever, user):
                self.retrieve = SecureRetrieve(
                    retriever=retriever,
                    user=user,
                    k=5
                )
                self.generate = dspy.ChainOfThought("context, question -> answer")

            def forward(self, question):
                context = self.retrieve(question).passages
                return self.generate(context=context, question=question)

        # Use with different users
        rag = SecureRAG(retriever=qdrant_retriever, user=alice_user)
        result = rag("What is our revenue?")

        # Switch user context
        rag.retrieve.set_user(bob_user)
        result = rag("What is our revenue?")
        ```
    """

    def __init__(
        self,
        retriever: Any = None,
        user: Dict[str, Any] = None,
        k: int = 3,
        **kwargs
    ):
        """
        Initialize secure retrieve module.

        Args:
            retriever: RAGGuard secure retriever instance
            user: User context for permission filtering
            k: Number of passages to retrieve
            **kwargs: Additional arguments
        """
        _check_dspy_available()

        super().__init__(k=k)

        if retriever is None:
            raise ValueError("retriever is required")
        if user is None:
            raise ValueError("user context is required")

        self._retriever = retriever
        self._user = user

    def forward(
        self,
        query_or_queries: Union[str, List[str]],
        k: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Retrieve passages for query/queries.

        Args:
            query_or_queries: Single query or list of queries
            k: Number of results per query
            **kwargs: Additional search parameters

        Returns:
            DSPy Prediction with passages
        """
        k = k or self.k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries

        all_passages = []

        for query in queries:
            try:
                results = self._retriever.search(
                    query=query,
                    user=self._user,
                    limit=k,
                    **kwargs
                )

                passages = self._extract_passages(results)
                all_passages.extend(passages)

            except Exception as e:
                logger.error(f"Retrieval failed for query '{query}': {e}")

        # Remove duplicates while preserving order
        seen = set()
        unique_passages = []
        for p in all_passages:
            if p not in seen:
                seen.add(p)
                unique_passages.append(p)

        return _Prediction(passages=unique_passages[:k])

    def _extract_passages(self, results: List[Any]) -> List[str]:
        """Extract text from results."""
        passages = []

        for result in results:
            if hasattr(result, 'payload'):
                text = (
                    result.payload.get('text') or
                    result.payload.get('content') or
                    str(result.payload)
                )
            elif isinstance(result, dict):
                text = (
                    result.get('text') or
                    result.get('content') or
                    result.get('metadata', {}).get('text', '') or
                    str(result)
                )
            else:
                text = str(result)

            passages.append(text)

        return passages

    def set_user(self, user: Dict[str, Any]) -> "SecureRetrieve":
        """Update user context."""
        self._user = user
        return self

    @property
    def user(self) -> Dict[str, Any]:
        """Get current user context."""
        return self._user


class SecureRAG(_Module if DSPY_AVAILABLE else object):  # type: ignore[misc]
    """
    Pre-built RAG module with secure retrieval.

    This is a convenience class providing a complete RAG pipeline
    with built-in permission filtering.

    Example:
        ```python
        from ragguard.integrations.dspy import SecureRAG

        # Create RAG module
        rag = SecureRAG(
            retriever=qdrant_retriever,
            user={"id": "alice", "department": "engineering"},
            k=5
        )

        # Query
        result = rag("What were our Q3 sales?")
        print(result.answer)
        print(result.context)  # Retrieved passages

        # Change user
        rag.set_user({"id": "bob", "department": "hr"})
        result = rag("What were our Q3 sales?")  # May get different results
        ```
    """

    def __init__(
        self,
        retriever: Any,
        user: Dict[str, Any],
        k: int = 5,
        signature: str = "context, question -> answer"
    ):
        """
        Initialize secure RAG module.

        Args:
            retriever: RAGGuard secure retriever
            user: User context for permissions
            k: Number of passages to retrieve
            signature: DSPy signature for generation
        """
        _check_dspy_available()

        super().__init__()

        self.retrieve = SecureRetrieve(
            retriever=retriever,
            user=user,
            k=k
        )
        self.generate = dspy.ChainOfThought(signature)
        self._user = user

    def forward(self, question: str) -> Any:
        """
        Answer a question using secure retrieval.

        Args:
            question: The question to answer

        Returns:
            DSPy Prediction with answer and context
        """
        # Retrieve relevant passages
        retrieval_result = self.retrieve(question)
        context = retrieval_result.passages

        # Generate answer
        if context:
            context_str = "\n\n".join(context)
            result = self.generate(context=context_str, question=question)
            result.context = context
        else:
            result = _Prediction(
                answer="I don't have access to documents that could answer this question.",
                context=[]
            )

        return result

    def set_user(self, user: Dict[str, Any]) -> "SecureRAG":
        """Update user context."""
        self._user = user
        self.retrieve.set_user(user)
        return self

    @property
    def user(self) -> Dict[str, Any]:
        """Get current user context."""
        return self._user


def configure_ragguard_rm(
    retriever: Any,
    user: Dict[str, Any],
    k: int = 5
) -> RAGGuardRM:
    """
    Configure DSPy to use RAGGuard as the retriever model.

    This is a convenience function that creates a RAGGuardRM and
    configures it as DSPy's default retriever.

    Args:
        retriever: RAGGuard secure retriever
        user: User context for permission filtering
        k: Default number of results

    Returns:
        The configured RAGGuardRM instance

    Example:
        ```python
        from ragguard.integrations.dspy import configure_ragguard_rm

        # Configure DSPy with RAGGuard
        rm = configure_ragguard_rm(
            retriever=qdrant_retriever,
            user={"id": "alice", "department": "engineering"}
        )

        # Now dspy.Retrieve() uses RAGGuard automatically
        retrieve = dspy.Retrieve(k=3)
        results = retrieve("quarterly report")
        ```
    """
    _check_dspy_available()

    rm = RAGGuardRM(
        retriever=retriever,
        user=user,
        k=k
    )

    dspy.settings.configure(rm=rm)

    return rm


__all__ = [
    "RAGGuardRM",
    "SecureRetrieve",
    "SecureRAG",
    "configure_ragguard_rm",
    "DSPY_AVAILABLE",
]
