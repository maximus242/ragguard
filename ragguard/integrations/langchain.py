"""
LangChain integration for RAGGuard.

Provides a LangChain-compatible retriever with permission-aware search.
"""

from typing import Any, Callable, Dict, List, Optional

from pydantic import ConfigDict

# Check if LangChain is available
try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Create dummy classes for type hints
    BaseRetriever = object  # type: ignore
    Document = object  # type: ignore
    CallbackManagerForRetrieverRun = object  # type: ignore


def _check_langchain_available():
    """Check if LangChain is installed."""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain integration requires langchain. "
            "Install with: pip install ragguard[langchain]"
        )

from ..audit.logger import AuditLogger
from ..exceptions import RetrieverError
from ..policy.models import Policy
from ..retrievers import QdrantSecureRetriever


class LangChainSecureRetriever(BaseRetriever):
    """
    LangChain-compatible retriever with permission-aware search.

    This retriever wraps RAGGuard's SecureRetriever and converts results
    to LangChain Document objects.

    Usage:
        ```python
        from ragguard.integrations.langchain import LangChainSecureRetriever

        retriever = LangChainSecureRetriever(
            qdrant_client=client,
            collection="documents",
            policy=policy,
            embedding_function=embeddings.embed_query
        )

        # Use in LangChain - pass user context in metadata
        docs = retriever.get_relevant_documents(
            "query text",
            user={"id": "alice@company.com", "roles": ["engineer"]}
        )

        # Or use with LangChain chains/agents
        # The user context must be set on the retriever instance
        retriever.set_user({"id": "alice@company.com", "roles": ["engineer"]})
        docs = retriever.get_relevant_documents("query text")
        ```
    """

    qdrant_client: Any
    collection: str
    policy: Policy
    embedding_function: Callable[[str], List[float]]
    audit_logger: Optional[AuditLogger] = None
    current_user: Optional[Dict[str, Any]] = None
    search_kwargs: Dict[str, Any] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(
        self,
        qdrant_client: Any,
        collection: str,
        policy: Policy,
        embedding_function: Callable[[str], List[float]],
        audit_logger: Optional[AuditLogger] = None,
        **kwargs
    ):
        """
        Initialize LangChain-compatible secure retriever.

        Args:
            qdrant_client: QdrantClient instance
            collection: Collection name
            policy: Access control policy
            embedding_function: Function to convert text to embeddings
            audit_logger: Optional audit logger
            **kwargs: Additional arguments passed to parent class
        """
        _check_langchain_available()

        super().__init__(  # type: ignore[call-arg]
            qdrant_client=qdrant_client,
            collection=collection,
            policy=policy,
            embedding_function=embedding_function,
            audit_logger=audit_logger,
            current_user=None,
            search_kwargs={},
            **kwargs
        )

        # Create the underlying RAGGuard retriever
        self._ragguard_retriever = QdrantSecureRetriever(
            client=qdrant_client,
            collection=collection,
            policy=policy,
            audit_logger=audit_logger,
            embed_fn=embedding_function
        )

    def set_user(self, user: Dict[str, Any]) -> "LangChainSecureRetriever":
        """
        Set the user context for subsequent retrievals.

        Args:
            user: User context dictionary

        Returns:
            Self for method chaining
        """
        self.current_user = user
        return self

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve documents relevant to a query with permission filtering.

        Args:
            query: Query string
            run_manager: Callback manager (not used)
            **kwargs: Additional arguments, including optional 'user' context

        Returns:
            List of LangChain Document objects
        """
        # Get user context from kwargs or use current_user
        user = kwargs.pop("user", None) or self.current_user

        if user is None:
            raise RetrieverError(
                "User context required. Either pass user={...} or call set_user() first."
            )

        # Merge search kwargs
        search_params = {**self.search_kwargs, **kwargs}
        limit = search_params.pop("k", 10)  # LangChain uses 'k' for limit

        # Execute permission-aware search using RAGGuard
        results = self._ragguard_retriever.search(
            query=query,
            user=user,
            limit=limit,
            **search_params
        )

        # Convert to LangChain Documents
        documents = []
        for result in results:
            # Handle both Qdrant ScoredPoint (old API) and QueryResponse (new API)
            if hasattr(result, 'payload'):
                # Old API: ScoredPoint
                payload = result.payload
                score = result.score
            elif hasattr(result, 'metadata'):
                # New API: QueryResponse
                payload = result.metadata or {}
                score = result.score
            else:
                # Fallback
                payload = {}
                score = 0.0

            # Extract text content
            page_content = payload.pop("text", "") or payload.pop("content", "") or str(payload)

            # Create LangChain Document
            doc = Document(
                page_content=page_content,
                metadata={
                    **payload,
                    "score": score,
                }
            )
            documents.append(doc)

        return documents


class LangChainPgvectorSecureRetriever(BaseRetriever):
    """
    LangChain-compatible retriever for pgvector with permission-aware search.

    Similar to LangChainSecureRetriever but for PostgreSQL with pgvector.
    """

    connection: Any
    table: str
    policy: Policy
    embedding_function: Callable[[str], List[float]]
    embedding_column: str = "embedding"
    text_column: str = "text"
    audit_logger: Optional[AuditLogger] = None
    current_user: Optional[Dict[str, Any]] = None
    search_kwargs: Dict[str, Any] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(
        self,
        connection: Any,
        table: str,
        policy: Policy,
        embedding_function: Callable[[str], List[float]],
        embedding_column: str = "embedding",
        text_column: str = "text",
        audit_logger: Optional[AuditLogger] = None,
        **kwargs
    ):
        """
        Initialize pgvector LangChain-compatible secure retriever.

        Args:
            connection: PostgreSQL connection
            table: Table name
            policy: Access control policy
            embedding_function: Function to convert text to embeddings
            embedding_column: Name of embedding column
            text_column: Name of text content column
            audit_logger: Optional audit logger
            **kwargs: Additional arguments
        """
        _check_langchain_available()

        super().__init__(  # type: ignore[call-arg]
            connection=connection,
            table=table,
            policy=policy,
            embedding_function=embedding_function,
            embedding_column=embedding_column,
            text_column=text_column,
            audit_logger=audit_logger,
            current_user=None,
            search_kwargs={},
            **kwargs
        )

        from ..retrievers import PgvectorSecureRetriever

        self._ragguard_retriever = PgvectorSecureRetriever(
            connection=connection,
            table=table,
            policy=policy,
            embedding_column=embedding_column,
            audit_logger=audit_logger,
            embed_fn=embedding_function
        )

    def set_user(self, user: Dict[str, Any]) -> "LangChainPgvectorSecureRetriever":
        """Set user context."""
        self.current_user = user
        return self

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs
    ) -> List[Document]:
        """Retrieve documents with permission filtering."""
        user = kwargs.pop("user", None) or self.current_user

        if user is None:
            raise RetrieverError(
                "User context required. Either pass user={...} or call set_user() first."
            )

        search_params = {**self.search_kwargs, **kwargs}
        limit = search_params.pop("k", 10)

        results = self._ragguard_retriever.search(
            query=query,
            user=user,
            limit=limit,
            **search_params
        )

        documents = []
        for result in results:
            text_content = result.get(self.text_column, "")
            metadata = {k: v for k, v in result.items() if k != self.text_column}

            doc = Document(
                page_content=text_content,
                metadata=metadata
            )
            documents.append(doc)

        return documents


class LangChainRetrieverWrapper(BaseRetriever):
    """
    Generic LangChain wrapper for any RAGGuard retriever.

    This wrapper can wrap any RAGGuard SecureRetriever and make it compatible
    with LangChain's retriever interface. It handles converting results to
    LangChain Document objects and managing user context.

    Usage:
        ```python
        from ragguard.integrations.langchain import LangChainRetrieverWrapper
        from ragguard.retrievers import ChromaDBSecureRetriever, PineconeSecureRetriever

        # Wrap any RAGGuard retriever
        chromadb_retriever = ChromaDBSecureRetriever(
            client=chroma_client,
            collection="documents",
            policy=policy,
            embed_fn=embeddings.embed_query
        )

        langchain_retriever = LangChainRetrieverWrapper(
            retriever=chromadb_retriever,
            text_key="text"  # Key in metadata containing document text
        )

        # Use in LangChain
        langchain_retriever.set_user({"id": "alice", "roles": ["engineer"]})
        docs = langchain_retriever.get_relevant_documents("query text")

        # Or pass user context directly
        docs = langchain_retriever.get_relevant_documents(
            "query text",
            user={"id": "alice", "roles": ["engineer"]}
        )
        ```

    Works with all RAGGuard retrievers:
        - ChromaDBSecureRetriever
        - PineconeSecureRetriever
        - MilvusSecureRetriever
        - WeaviateSecureRetriever
        - ElasticsearchSecureRetriever
        - FAISSSecureRetriever
        - AzureSearchSecureRetriever
        - QdrantSecureRetriever
        - PgvectorSecureRetriever
    """

    retriever: Any  # The underlying RAGGuard retriever
    text_key: str = "text"  # Key in result containing document text
    content_keys: List[str] = ["text", "content", "page_content", "body"]  # Fallback keys
    current_user: Optional[Dict[str, Any]] = None
    search_kwargs: Dict[str, Any] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(
        self,
        retriever: Any,
        text_key: str = "text",
        content_keys: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize generic LangChain wrapper.

        Args:
            retriever: Any RAGGuard SecureRetriever instance
            text_key: Primary key in result metadata containing document text
            content_keys: Fallback keys to try if text_key not found
            **kwargs: Additional arguments passed to parent class
        """
        _check_langchain_available()

        if content_keys is None:
            content_keys = ["text", "content", "page_content", "body"]

        super().__init__(  # type: ignore[call-arg]
            retriever=retriever,
            text_key=text_key,
            content_keys=content_keys,
            current_user=None,
            search_kwargs={},
            **kwargs
        )

    def set_user(self, user: Dict[str, Any]) -> "LangChainRetrieverWrapper":
        """
        Set the user context for subsequent retrievals.

        Args:
            user: User context dictionary (e.g., {"id": "alice", "roles": ["admin"]})

        Returns:
            Self for method chaining
        """
        self.current_user = user
        return self

    def _extract_text_content(self, result: Any) -> tuple[str, Dict[str, Any]]:
        """
        Extract text content and metadata from a search result.

        Handles various result formats from different RAGGuard retrievers:
        - Qdrant ScoredPoint (has .payload and .score)
        - Dict results (from most retrievers)
        - Objects with metadata attribute

        Args:
            result: A single search result

        Returns:
            Tuple of (text_content, metadata_dict)
        """
        # Handle Qdrant ScoredPoint
        if hasattr(result, 'payload'):
            metadata = dict(result.payload) if result.payload else {}
            score = getattr(result, 'score', 0.0)
            metadata['score'] = score
        # Handle dict results
        elif isinstance(result, dict):
            metadata = dict(result)
            score = metadata.pop('score', metadata.pop('_score', 0.0))
            metadata['score'] = score
        # Handle objects with metadata attribute
        elif hasattr(result, 'metadata'):
            metadata = dict(result.metadata) if result.metadata else {}
            score = getattr(result, 'score', 0.0)
            metadata['score'] = score
        else:
            # Fallback: try to convert to dict
            try:
                metadata = dict(result)
            except (TypeError, ValueError):
                metadata = {"raw_result": str(result)}
            metadata['score'] = 0.0

        # Extract text content
        text_content = ""

        # Try primary text_key first
        if self.text_key in metadata:
            text_content = str(metadata.pop(self.text_key, ""))
        else:
            # Try fallback content keys
            for key in self.content_keys:
                if key in metadata:
                    text_content = str(metadata.pop(key, ""))
                    break

        # If still no content, use string representation
        if not text_content:
            text_content = str(metadata.get('raw_result', ''))

        return text_content, metadata

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve documents relevant to a query with permission filtering.

        Args:
            query: Query string
            run_manager: Callback manager (not used)
            **kwargs: Additional arguments, including optional 'user' context

        Returns:
            List of LangChain Document objects
        """
        # Get user context from kwargs or use current_user
        user = kwargs.pop("user", None) or self.current_user

        if user is None:
            raise RetrieverError(
                "User context required for permission filtering. "
                "Either pass user={...} to get_relevant_documents() or call set_user() first."
            )

        # Merge search kwargs
        search_params = {**self.search_kwargs, **kwargs}
        limit = search_params.pop("k", search_params.pop("limit", 10))

        # Execute permission-aware search using the underlying RAGGuard retriever
        results = self.retriever.search(
            query=query,
            user=user,
            limit=limit,
            **search_params
        )

        # Convert results to LangChain Documents
        documents = []
        for result in results:
            text_content, metadata = self._extract_text_content(result)

            doc = Document(
                page_content=text_content,
                metadata=metadata
            )
            documents.append(doc)

        return documents


# Convenience function for wrapping retrievers
def wrap_retriever(
    retriever: Any,
    text_key: str = "text",
    **kwargs
) -> LangChainRetrieverWrapper:
    """
    Wrap any RAGGuard retriever for use with LangChain.

    This is a convenience function to quickly wrap a RAGGuard retriever.

    Args:
        retriever: Any RAGGuard SecureRetriever instance
        text_key: Key in result metadata containing document text
        **kwargs: Additional arguments

    Returns:
        LangChainRetrieverWrapper instance

    Example:
        ```python
        from ragguard.retrievers import ChromaDBSecureRetriever
        from ragguard.integrations.langchain import wrap_retriever

        # Create RAGGuard retriever
        secure_retriever = ChromaDBSecureRetriever(
            client=chroma_client,
            collection="docs",
            policy=policy,
            embed_fn=embed_fn
        )

        # Wrap for LangChain
        langchain_retriever = wrap_retriever(secure_retriever)

        # Use with user context
        langchain_retriever.set_user({"id": "alice"})
        docs = langchain_retriever.get_relevant_documents("search query")
        ```
    """
    return LangChainRetrieverWrapper(
        retriever=retriever,
        text_key=text_key,
        **kwargs
    )
