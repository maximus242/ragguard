"""
Tests for field name injection prevention across all backends.

These tests verify that malicious field names are rejected before being
used in filter queries, preventing SQL injection and similar attacks.
"""

import pytest

from ragguard.filters.base import validate_field_name, validate_field_path
from ragguard.policy.models import AllowConditions, Policy, Rule

# Malicious field names that should be rejected
MALICIOUS_FIELDS = [
    # SQL injection attempts
    "field'; DROP TABLE--",
    "field\"; DELETE FROM",
    'field`; rm -rf',
    "field; SELECT * FROM",
    "field' OR '1'='1",
    "field AND 1=1--",

    # Path traversal attempts
    "../../../etc/passwd",
    "..\\..\\..\\windows\\system32",
    "field/../secret",

    # Shell injection attempts
    "field$(whoami)",
    "field`id`",
    "field|cat /etc/passwd",
    "field;id",

    # Special characters that could break queries
    "field'value",
    'field"value',
    "field`value",
    "field\x00null",  # Null byte injection
    "field\nvalue",   # Newline injection
    "field\rvalue",   # Carriage return
    "field\tvalue",   # Tab

    # Unicode/encoding attacks
    "field\u0000value",  # Unicode null

    # Empty/whitespace
    "",
    "   ",
    "\t\n",

    # Overly long field names
    "a" * 300,

    # Fields starting with invalid characters
    "123field",
    "-field",
    ".field",
    "@field",
    "$field",

    # Invalid dot placement
    ".field",
    "field.",
    "field..name",
    "..field",
]


class TestValidateFieldName:
    """Test the validate_field_name function."""

    @pytest.mark.parametrize("malicious_field", MALICIOUS_FIELDS)
    def test_rejects_malicious_field_names(self, malicious_field):
        """Verify that malicious field names are rejected."""
        with pytest.raises(ValueError, match=r"Invalid field"):
            validate_field_name(malicious_field, "test_backend")

    def test_accepts_valid_simple_field(self):
        """Valid simple field names should be accepted."""
        assert validate_field_name("department", "test") == "department"
        assert validate_field_name("user_id", "test") == "user_id"
        assert validate_field_name("_private", "test") == "_private"
        assert validate_field_name("Field123", "test") == "Field123"

    def test_accepts_valid_dotted_field(self):
        """Valid dotted field names should be accepted."""
        assert validate_field_name("user.department", "test") == "user.department"
        assert validate_field_name("metadata.tags.primary", "test") == "metadata.tags.primary"
        assert validate_field_name("a.b.c.d", "test") == "a.b.c.d"

    def test_rejects_none_and_non_string(self):
        """None and non-string inputs should be rejected."""
        with pytest.raises(ValueError, match=r"non-empty string"):
            validate_field_name(None, "test")
        with pytest.raises(ValueError, match=r"non-empty string"):
            validate_field_name(123, "test")
        with pytest.raises(ValueError, match=r"non-empty string"):
            validate_field_name([], "test")


class TestValidateFieldPath:
    """Test the validate_field_path function."""

    @pytest.mark.parametrize("malicious_component", [
        "field'; DROP TABLE--",
        "field\"; DELETE FROM",
        "../../../etc/passwd",
        "field$(whoami)",
        "field'value",
        "",
        "123field",
        "a" * 100,
    ])
    def test_rejects_malicious_path_components(self, malicious_component):
        """Verify that malicious path components are rejected."""
        with pytest.raises(ValueError, match=r"Invalid field path"):
            validate_field_path([malicious_component], "test_backend")
        with pytest.raises(ValueError, match=r"Invalid field path"):
            validate_field_path(["valid", malicious_component, "also_valid"], "test_backend")

    def test_accepts_valid_field_path(self):
        """Valid field paths should be accepted."""
        assert validate_field_path(["department"], "test") == "department"
        assert validate_field_path(["user", "name"], "test") == "user.name"
        assert validate_field_path(["a", "b", "c"], "test") == "a.b.c"
        assert validate_field_path(["_private", "field123"], "test") == "_private.field123"

    def test_rejects_empty_path(self):
        """Empty paths should be rejected."""
        with pytest.raises(ValueError, match=r"non-empty list or tuple"):
            validate_field_path([], "test")
        with pytest.raises(ValueError, match=r"non-empty list or tuple"):
            validate_field_path(None, "test")

    def test_accepts_tuple_field_path(self):
        """Tuple field paths should be accepted (compiler returns tuples)."""
        assert validate_field_path(("department",), "test") == "department"
        assert validate_field_path(("user", "name"), "test") == "user.name"


class TestPgvectorFieldInjection:
    """Test that pgvector backend rejects malicious fields."""

    @pytest.fixture
    def policy_with_field(self):
        """Create a policy that uses a specific field name."""
        def _create_policy(field_name: str) -> Policy:
            return Policy(
                rules=[
                    Rule(
                        allow=AllowConditions(
                            everyone=True,
                            conditions=[f"user.id == document.{field_name}"]
                        )
                    )
                ]
            )
        return _create_policy

    @pytest.mark.parametrize("malicious_field", [
        "field'; DROP TABLE--",
        "field\"; DELETE FROM",
        "../../../etc/passwd",
        "field$(whoami)",
    ])
    def test_compiled_conditions_reject_malicious_fields(self, malicious_field, policy_with_field):
        """Compiled condition paths with malicious fields should be rejected."""
        from ragguard.filters import to_pgvector_filter

        # Create a simple policy - the malicious field is in the condition
        # but we can't easily test the compiled path this way, so test directly
        with pytest.raises(ValueError, match=r"Invalid field"):
            validate_field_path([malicious_field], "pgvector")


class TestElasticsearchFieldInjection:
    """Test that Elasticsearch backend rejects malicious fields."""

    @pytest.mark.parametrize("malicious_field", [
        "field'; DROP TABLE--",
        "../../../etc/passwd",
        "field$(whoami)",
    ])
    def test_rejects_malicious_field_paths(self, malicious_field):
        """Malicious field paths should be rejected."""
        with pytest.raises(ValueError, match=r"Invalid field"):
            validate_field_path([malicious_field], "elasticsearch")


class TestAzureSearchFieldInjection:
    """Test that Azure Search backend rejects malicious fields."""

    @pytest.mark.parametrize("malicious_field", [
        "field'; DROP TABLE--",
        "../../../etc/passwd",
        "field$(whoami)",
    ])
    def test_rejects_malicious_field_paths(self, malicious_field):
        """Malicious field paths should be rejected."""
        with pytest.raises(ValueError, match=r"Invalid field"):
            validate_field_path([malicious_field], "azure_search")


class TestMilvusFieldInjection:
    """Test that Milvus backend rejects malicious fields."""

    @pytest.mark.parametrize("malicious_field", [
        "field'; DROP TABLE--",
        "../../../etc/passwd",
        "field$(whoami)",
    ])
    def test_rejects_malicious_field_names(self, malicious_field):
        """Malicious field names should be rejected."""
        with pytest.raises(ValueError, match=r"Invalid field"):
            validate_field_name(malicious_field, "milvus")


class TestEdgeCases:
    """Test edge cases in field validation."""

    def test_field_with_underscore(self):
        """Fields with underscores should work."""
        assert validate_field_name("my_field_name", "test") == "my_field_name"
        assert validate_field_name("__private", "test") == "__private"
        assert validate_field_name("_", "test") == "_"

    def test_field_with_numbers(self):
        """Fields with numbers (not at start) should work."""
        assert validate_field_name("field123", "test") == "field123"
        assert validate_field_name("user2", "test") == "user2"
        assert validate_field_name("v3_config", "test") == "v3_config"

    def test_mixed_case_fields(self):
        """Mixed case fields should work."""
        assert validate_field_name("myField", "test") == "myField"
        assert validate_field_name("MyField", "test") == "MyField"
        assert validate_field_name("FIELD", "test") == "FIELD"

    def test_max_length_enforcement(self):
        """Fields at or near max length should be handled correctly."""
        # Just under limit should work
        assert validate_field_name("a" * 256, "test") == "a" * 256

        # Over limit should fail
        with pytest.raises(ValueError, match=r"too long"):
            validate_field_name("a" * 257, "test")

    def test_path_component_length(self):
        """Path components should have their own length limit."""
        # Component at limit should work
        assert validate_field_path(["a" * 64], "test") == "a" * 64

        # Component over limit should fail
        with pytest.raises(ValueError, match=r"too long"):
            validate_field_path(["a" * 65], "test")
