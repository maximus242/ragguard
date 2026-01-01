"""
Example: Improved error messages with context.

This example demonstrates how RAGGuard provides helpful error messages
with context, suggestions, and actionable advice.
"""

from ragguard import Policy
from ragguard.policy.engine import PolicyEngine
from ragguard.retrievers import QdrantSecureRetriever
from ragguard.validation import InputValidator, ValidationConfig
from ragguard.exceptions import PolicyEvaluationError, RetrieverError


def demo_unsupported_backend_error():
    """Demonstrate improved error message for unsupported backend."""
    print("=" * 80)
    print("DEMO 1: Unsupported Backend Error")
    print("=" * 80)

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "all", "allow": {"everyone": True}}],
        "default": "deny"
    })

    engine = PolicyEngine(policy)
    user = {"id": "alice"}

    try:
        # Try to generate filter for unsupported backend
        engine.to_filter(user, "my_custom_db")
    except PolicyEvaluationError as e:
        print(str(e))
        print("\n✓ Notice how the error provides:")
        print("  - What was attempted")
        print("  - List of supported options")
        print("  - Suggestions for how to fix")
        print("  - Installation commands")


def demo_empty_user_context_error():
    """Demonstrate improved error message for empty user context."""
    print("\n" + "=" * 80)
    print("DEMO 2: Empty User Context Error")
    print("=" * 80)

    validator = InputValidator(ValidationConfig())

    try:
        # Try to validate empty user context
        validator.validate_user_context({})
    except RetrieverError as e:
        print(str(e))
        print("\n✓ Notice how the error provides:")
        print("  - Clear explanation of what's wrong")
        print("  - Example of correct usage")
        print("  - Suggestions for what to include")


def demo_error_context_builder():
    """Demonstrate using ErrorContext to create custom error messages."""
    print("\n" + "=" * 80)
    print("DEMO 3: Custom Error with ErrorContext")
    print("=" * 80)

    from ragguard.errors import ErrorContext

    # Build a custom error message
    error = (
        ErrorContext("Configuration Error")
        .with_attempted_value("invalid_config.yaml")
        .with_valid_options(["config.yaml", "config.json", "config.toml"])
        .with_suggestion("Check that the file exists")
        .with_suggestion("Verify the file format is correct")
        .with_suggestion("Ensure you have read permissions")
        .with_context("current_directory", "/home/user/app")
        .with_context("expected_location", "/home/user/app/config")
        .with_doc_link("https://github.com/maximus242/ragguard#configuration")
        .build()
    )

    print(error)
    print("\n✓ ErrorContext provides a fluent API for building")
    print("  comprehensive error messages with context!")


def compare_before_after():
    """Compare old vs new error messages."""
    print("\n" + "=" * 80)
    print("DEMO 4: Before vs After Comparison")
    print("=" * 80)

    print("\n❌ OLD ERROR MESSAGE (generic):")
    print("  RetrieverError: Unsupported backend: elasticsearch")
    print("  ↑ Not helpful! What backends ARE supported?")
    print("     How do I install the one I need?")

    print("\n✅ NEW ERROR MESSAGE (with context):")
    print("""
======================================================================
ERROR: Unsupported Backend
======================================================================

Attempted: 'elasticsearch'

Supported options (10):
  - azure_search
  - chromadb
  - elasticsearch
  - faiss
  - milvus
  - opensearch
  - pgvector
  - pinecone
  - qdrant
  - weaviate

Context:
  operation: filter generation

How to fix:
  1. Check if 'elasticsearch' is the correct spelling (case-sensitive)
  2. Verify that you have the required backend library installed
  3. Install with: pip install ragguard[elasticsearch]

Documentation: https://github.com/maximus242/ragguard#backends
======================================================================
    """)

    print("\n✓ Much better! The new message provides:")
    print("  - Complete list of supported backends")
    print("  - Exact installation command")
    print("  - Link to documentation")
    print("  - Context about what operation failed")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RAGGuard: Improved Error Messages Demo")
    print("=" * 80)
    print("\nRAGGuard now provides helpful, actionable error messages")
    print("that include context, suggestions, and documentation links.")
    print("\nLet's see some examples...")

    try:
        demo_unsupported_backend_error()
    except Exception as e:
        print(f"Demo error: {e}")

    try:
        demo_empty_user_context_error()
    except Exception as e:
        print(f"Demo error: {e}")

    demo_error_context_builder()
    compare_before_after()

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("\nBetter error messages help developers:")
    print("  1. Understand what went wrong quickly")
    print("  2. Know exactly how to fix it")
    print("  3. Avoid common mistakes")
    print("  4. Reduce debugging time")
    print("\nUse ErrorContext in your own code for consistent")
    print("error messages throughout your application!")
    print()
