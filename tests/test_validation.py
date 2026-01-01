"""
Tests for input validation.

Tests that user and document contexts are properly validated to prevent:
- DOS attacks
- Field name injection
- Type confusion
- Invalid data
"""

import pytest

from ragguard.exceptions import RetrieverError


def test_valid_user_context():
    """Test that valid user context passes validation."""
    from ragguard import validate_user

    user = {"id": "alice", "department": "engineering", "roles": ["admin"]}
    result = validate_user(user)
    # Validation returns None on success (no exception raised)
    assert result is None


def test_valid_document_context():
    """Test that valid document context passes validation."""
    from ragguard import validate_document

    doc = {"id": "doc1", "department": "engineering", "owner": "alice"}
    result = validate_document(doc)
    assert result is None


def test_empty_user_context():
    """Test that empty user context fails validation."""
    from ragguard import validate_user

    with pytest.raises(RetrieverError, match="Empty User Context"):
        validate_user({})


def test_empty_document_context():
    """Test that empty document context is allowed."""
    from ragguard import validate_document

    result = validate_document({})  # Should not raise (for "everyone" policies)
    assert result is None


def test_non_dict_user_context():
    """Test that non-dict user context fails validation."""
    from ragguard import validate_user

    with pytest.raises(RetrieverError, match="User context must be a dictionary"):
        validate_user("alice")  # type: ignore


def test_non_dict_document_context():
    """Test that non-dict document context fails validation."""
    from ragguard import validate_document

    with pytest.raises(RetrieverError, match="Document context must be a dictionary"):
        validate_document(["doc1"])  # type: ignore


def test_dangerous_field_name_double_underscore():
    """Test that double underscore field names are rejected."""
    from ragguard import validate_user

    user = {"id": "alice", "__proto__": "malicious"}
    with pytest.raises(RetrieverError, match="Invalid field name.*contains dangerous pattern"):
        validate_user(user)


def test_dangerous_field_name_dollar_sign():
    """Test that dollar sign field names are rejected (MongoDB injection)."""
    from ragguard import validate_user

    user = {"id": "alice", "$where": "malicious"}
    with pytest.raises(RetrieverError, match="Invalid field name.*contains dangerous pattern"):
        validate_user(user)


def test_dangerous_field_name_semicolon():
    """Test that field names with semicolons are rejected (SQL injection)."""
    from ragguard import validate_user

    user = {"id": "alice", "name; DROP TABLE users;": "malicious"}
    with pytest.raises(RetrieverError, match="Invalid field name.*contains dangerous pattern"):
        validate_user(user)


def test_dangerous_field_name_script():
    """Test that field names with <script are rejected (XSS)."""
    from ragguard import validate_user

    user = {"id": "alice", "<script>alert('xss')</script>": "malicious"}
    with pytest.raises(RetrieverError, match="Invalid field name.*contains dangerous pattern"):
        validate_user(user)


def test_null_byte_in_string():
    """Test that null bytes in strings are rejected."""
    from ragguard import validate_user

    user = {"id": "alice\x00admin"}
    with pytest.raises(RetrieverError, match="contains null byte"):
        validate_user(user)


def test_max_string_length():
    """Test that very long strings are rejected."""
    from ragguard import ValidationConfig, validate_user

    user = {"id": "alice", "bio": "x" * 100000}  # 100k characters

    # Should fail with default config
    with pytest.raises(RetrieverError, match="string length.*exceeds maximum"):
        validate_user(user)


def test_max_string_length_custom():
    """Test that custom max string length works."""
    from ragguard import ValidationConfig, validate_user

    config = ValidationConfig(max_string_length=50)
    user = {"id": "alice", "bio": "x" * 100}

    with pytest.raises(RetrieverError, match="string length.*exceeds maximum"):
        validate_user(user, config)


def test_max_dict_size():
    """Test that dictionaries with too many fields are rejected."""
    from ragguard import ValidationConfig, validate_user

    config = ValidationConfig(max_dict_size=10)
    user = {f"field{i}": i for i in range(50)}

    with pytest.raises(RetrieverError, match="has 50 fields, maximum is 10"):
        validate_user(user, config)


def test_max_array_length():
    """Test that very long arrays are rejected."""
    from ragguard import ValidationConfig, validate_user

    config = ValidationConfig(max_array_length=100)
    user = {"id": "alice", "roles": list(range(500))}

    with pytest.raises(RetrieverError, match="array length.*exceeds maximum"):
        validate_user(user, config)


def test_max_nesting_depth():
    """Test that deeply nested objects are rejected (DOS protection)."""
    from ragguard import ValidationConfig, validate_user

    config = ValidationConfig(max_nesting_depth=3)

    # Create deeply nested object
    user = {"level1": {"level2": {"level3": {"level4": {"level5": "too deep"}}}}}

    with pytest.raises(RetrieverError, match="exceeds maximum nesting depth"):
        validate_user(user, config)


def test_shallow_nesting():
    """Test that shallow nesting is allowed."""
    from ragguard import validate_user

    user = {
        "id": "alice",
        "org": {
            "id": "org123",
            "department": {
                "id": "eng",
                "name": "Engineering"
            }
        }
    }

    result = validate_user(user)
    assert result is None  # Validation passes


def test_none_values_allowed():
    """Test that None values are allowed by default."""
    from ragguard import validate_user

    user = {"id": "alice", "manager": None}
    result = validate_user(user)
    assert result is None  # Validation passes


def test_none_values_disallowed():
    """Test that None values can be disallowed."""
    from ragguard import ValidationConfig, validate_user

    config = ValidationConfig(allow_none_values=False)
    user = {"id": "alice", "manager": None}

    with pytest.raises(RetrieverError, match="has None value but None values are not allowed"):
        validate_user(user, config)


def test_unsupported_types():
    """Test that unsupported types are rejected."""
    from ragguard import validate_user

    # Object/class instance
    user = {"id": "alice", "callback": lambda x: x}
    with pytest.raises(RetrieverError, match="unsupported type"):
        validate_user(user)


def test_valid_types():
    """Test that all supported types are allowed."""
    from ragguard import validate_user

    user = {
        "id": "alice",
        "age": 30,
        "rating": 4.5,
        "active": True,
        "roles": ["admin", "user"],
        "metadata": {"key": "value"},
        "manager": None
    }

    result = validate_user(user)
    assert result is None


def test_field_name_with_dots():
    """Test that field names with dots are allowed."""
    from ragguard import validate_user

    user = {"id": "alice", "org.id": "org123"}
    result = validate_user(user)
    assert result is None


def test_field_name_with_hyphens():
    """Test that field names with hyphens are allowed."""
    from ragguard import validate_user

    user = {"id": "alice", "first-name": "Alice"}
    result = validate_user(user)
    assert result is None


def test_field_name_with_underscores():
    """Test that field names with underscores are allowed."""
    from ragguard import validate_user

    user = {"id": "alice", "first_name": "Alice"}
    result = validate_user(user)
    assert result is None


def test_field_name_empty_string():
    """Test that empty string field names are rejected."""
    from ragguard import validate_user

    user = {"": "value"}
    with pytest.raises(RetrieverError, match="field name cannot be empty"):
        validate_user(user)


def test_field_name_special_characters():
    """Test that field names with special characters are rejected."""
    from ragguard import validate_user

    user = {"id": "alice", "field!@#": "value"}
    with pytest.raises(RetrieverError, match="must contain only letters, numbers, underscores, hyphens, and dots"):
        validate_user(user)


def test_non_strict_field_names():
    """Test that non-strict mode allows more field names."""
    from ragguard import ValidationConfig, validate_user

    config = ValidationConfig(strict_field_names=False)

    # This would normally fail strict validation
    user = {"id": "alice", "field!@#": "value"}

    # Should still check for dangerous patterns like __proto__
    user_with_proto = {"__proto__": "bad"}
    with pytest.raises(RetrieverError):
        validate_user(user_with_proto, config)


def test_retriever_validates_user_context():
    """Test that retriever validates user context during search."""
    pytest.importorskip("chromadb", exc_type=ImportError)
    from unittest.mock import Mock

    from ragguard import ChromaDBSecureRetriever, Policy

    mock_collection = Mock()
    mock_collection.name = "test"

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "all", "allow": {"everyone": True}}],
        "default": "deny"
    })

    retriever = ChromaDBSecureRetriever(
        collection=mock_collection,
        policy=policy,
        embed_fn=lambda x: [0.1, 0.2, 0.3]
    )

    # Valid user should work
    valid_user = {"id": "alice"}
    # Don't actually search (mock won't work), just validate the validation runs

    # Invalid user should fail
    invalid_user = {}
    with pytest.raises(RetrieverError, match="Empty User Context"):
        retriever.search("test query", invalid_user)


def test_retriever_validation_can_be_disabled():
    """Test that validation can be disabled in retriever."""
    from unittest.mock import Mock

    from ragguard import InputValidator, Policy, ValidationConfig
    from ragguard.retrievers.base import BaseSecureRetriever

    # Create a minimal mock retriever for testing
    class MockRetriever(BaseSecureRetriever):
        @property
        def backend_name(self) -> str:
            return "chromadb"  # Use a real backend to avoid filter builder errors

        def _execute_search(self, query, filter, limit, **kwargs):
            return []

        def _check_backend_health(self) -> bool:
            return True

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{"name": "all", "allow": {"everyone": True}}],
        "default": "deny"
    })

    # Create retriever with validation disabled
    retriever = MockRetriever(
        client=Mock(),
        collection="test",
        policy=policy,
        embed_fn=lambda x: [0.1, 0.2, 0.3],
        enable_validation=False
    )

    # Empty user should work when validation is disabled
    empty_user = {}
    results = retriever.search("test query", empty_user)
    assert results == []  # Mock returns empty list


def test_nested_arrays():
    """Test validation of nested arrays."""
    from ragguard import validate_user

    user = {
        "id": "alice",
        "roles": ["admin", "user"],
        "permissions": [
            {"resource": "docs", "actions": ["read", "write"]},
            {"resource": "users", "actions": ["read"]}
        ]
    }

    result = validate_user(user)
    assert result is None  # Validation passes


def test_complex_valid_structure():
    """Test validation of complex but valid structure."""
    from ragguard import validate_user

    user = {
        "id": "alice",
        "email": "alice@example.com",
        "age": 30,
        "active": True,
        "roles": ["admin", "developer"],
        "department": {
            "id": "eng",
            "name": "Engineering",
            "location": "Building A"
        },
        "projects": [
            {"id": "proj1", "name": "Project Alpha"},
            {"id": "proj2", "name": "Project Beta"}
        ],
        "metadata": {
            "created_at": "2024-01-01",
            "updated_at": "2024-01-15"
        }
    }

    result = validate_user(user)
    assert result is None  # Validation passes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
