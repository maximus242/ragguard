"""Tests for ragguard.utils module."""

import pytest

from ragguard.utils import (
    deep_merge,
    format_duration_ms,
    get_nested_value,
    is_document_field,
    is_user_field,
    is_valid_policy_version,
    parse_list_literal,
    parse_literal_value,
    sanitize_field_name,
    secure_compare,
    secure_contains,
    set_nested_value,
    strip_document_prefix,
    strip_field_prefix,
    strip_user_prefix,
    truncate_string,
)


class TestStripPrefixes:
    """Tests for prefix stripping functions."""

    def test_strip_user_prefix_with_prefix(self):
        assert strip_user_prefix("user.department") == "department"
        assert strip_user_prefix("user.roles") == "roles"
        assert strip_user_prefix("user.profile.name") == "profile.name"

    def test_strip_user_prefix_without_prefix(self):
        assert strip_user_prefix("department") == "department"
        assert strip_user_prefix("document.category") == "document.category"

    def test_strip_document_prefix_with_prefix(self):
        assert strip_document_prefix("document.category") == "category"
        assert strip_document_prefix("document.metadata.tags") == "metadata.tags"

    def test_strip_document_prefix_without_prefix(self):
        assert strip_document_prefix("category") == "category"
        assert strip_document_prefix("user.roles") == "user.roles"

    def test_strip_field_prefix_user(self):
        field, prefix_type = strip_field_prefix("user.department")
        assert field == "department"
        assert prefix_type == "user"

    def test_strip_field_prefix_document(self):
        field, prefix_type = strip_field_prefix("document.category")
        assert field == "category"
        assert prefix_type == "document"

    def test_strip_field_prefix_unknown(self):
        field, prefix_type = strip_field_prefix("other_field")
        assert field == "other_field"
        assert prefix_type == "unknown"


class TestFieldChecks:
    """Tests for field type checking functions."""

    def test_is_user_field(self):
        assert is_user_field("user.department") is True
        assert is_user_field("user.roles") is True
        assert is_user_field("document.category") is False
        assert is_user_field("other") is False

    def test_is_document_field(self):
        assert is_document_field("document.category") is True
        assert is_document_field("document.metadata.tags") is True
        assert is_document_field("user.roles") is False
        assert is_document_field("other") is False


class TestNestedValue:
    """Tests for nested value access functions."""

    def test_get_nested_value_simple(self):
        data = {"name": "Alice", "age": 30}
        assert get_nested_value(data, "name") == "Alice"
        assert get_nested_value(data, "age") == 30

    def test_get_nested_value_nested(self):
        data = {"user": {"profile": {"name": "Alice", "email": "alice@example.com"}}}
        assert get_nested_value(data, "user.profile.name") == "Alice"
        assert get_nested_value(data, "user.profile.email") == "alice@example.com"

    def test_get_nested_value_missing(self):
        data = {"user": {"name": "Alice"}}
        assert get_nested_value(data, "user.email") is None
        assert get_nested_value(data, "user.email", "default") == "default"
        assert get_nested_value(data, "missing.path", "default") == "default"

    def test_get_nested_value_empty(self):
        assert get_nested_value({}, "any.path") is None
        assert get_nested_value(None, "any.path") is None
        assert get_nested_value({"key": "value"}, "") is None

    def test_set_nested_value_simple(self):
        data = {}
        set_nested_value(data, "name", "Alice")
        assert data == {"name": "Alice"}

    def test_set_nested_value_nested(self):
        data = {}
        set_nested_value(data, "user.profile.name", "Alice")
        assert data == {"user": {"profile": {"name": "Alice"}}}

    def test_set_nested_value_existing(self):
        data = {"user": {"name": "Bob"}}
        set_nested_value(data, "user.email", "bob@example.com")
        assert data == {"user": {"name": "Bob", "email": "bob@example.com"}}


class TestParseLiteralValue:
    """Tests for literal value parsing."""

    def test_parse_string_single_quotes(self):
        assert parse_literal_value("'hello'") == "hello"
        assert parse_literal_value("'world'") == "world"

    def test_parse_string_double_quotes(self):
        assert parse_literal_value('"hello"') == "hello"
        assert parse_literal_value('"world"') == "world"

    def test_parse_numbers(self):
        assert parse_literal_value("42") == 42
        assert parse_literal_value("-17") == -17
        assert parse_literal_value("3.14") == 3.14
        assert parse_literal_value("-0.5") == -0.5

    def test_parse_booleans(self):
        assert parse_literal_value("true") is True
        assert parse_literal_value("True") is True
        assert parse_literal_value("TRUE") is True
        assert parse_literal_value("false") is False
        assert parse_literal_value("False") is False

    def test_parse_none(self):
        assert parse_literal_value("none") is None
        assert parse_literal_value("None") is None
        assert parse_literal_value("null") is None
        assert parse_literal_value("NULL") is None

    def test_parse_list(self):
        assert parse_literal_value("['a', 'b', 'c']") == ["a", "b", "c"]
        assert parse_literal_value("[1, 2, 3]") == [1, 2, 3]

    def test_parse_unknown(self):
        # Unknown values returned as-is (field references)
        assert parse_literal_value("some_field") == "some_field"


class TestParseListLiteral:
    """Tests for list literal parsing."""

    def test_parse_string_list(self):
        assert parse_list_literal("['cs.AI', 'cs.LG']") == ["cs.AI", "cs.LG"]
        assert parse_list_literal('["a", "b", "c"]') == ["a", "b", "c"]

    def test_parse_number_list(self):
        assert parse_list_literal("[1, 2, 3]") == [1, 2, 3]
        assert parse_list_literal("[1.5, 2.5, 3.5]") == [1.5, 2.5, 3.5]

    def test_parse_empty_list(self):
        assert parse_list_literal("[]") == []
        assert parse_list_literal("[  ]") == []

    def test_parse_invalid_list(self):
        assert parse_list_literal("not a list") == []


class TestSanitizeFieldName:
    """Tests for field name sanitization."""

    def test_valid_field_names(self):
        assert sanitize_field_name("department") == "department"
        assert sanitize_field_name("user_id") == "user_id"
        assert sanitize_field_name("metadata.tags") == "metadata.tags"
        assert sanitize_field_name("field-name") == "field-name"

    def test_invalid_characters(self):
        with pytest.raises(ValueError):
            sanitize_field_name("field;name")
        with pytest.raises(ValueError):
            sanitize_field_name("field name")  # spaces
        with pytest.raises(ValueError):
            sanitize_field_name("field'name")

    def test_sql_injection_patterns(self):
        with pytest.raises(ValueError):
            sanitize_field_name("field--comment")
        with pytest.raises(ValueError):
            sanitize_field_name("field;drop")


class TestTruncateString:
    """Tests for string truncation."""

    def test_no_truncation_needed(self):
        assert truncate_string("hello", 10) == "hello"
        assert truncate_string("test", 4) == "test"

    def test_truncation(self):
        assert truncate_string("Hello World", 8) == "Hello..."
        assert truncate_string("Hello World", 5) == "He..."

    def test_custom_suffix(self):
        assert truncate_string("Hello World", 8, "~") == "Hello W~"


class TestFormatDuration:
    """Tests for duration formatting."""

    def test_format_duration_ms(self):
        assert format_duration_ms(0.12345) == "123.45ms"
        assert format_duration_ms(0.001) == "1.00ms"
        assert format_duration_ms(1.5) == "1500.00ms"


class TestPolicyVersion:
    """Tests for policy version validation."""

    def test_valid_versions(self):
        assert is_valid_policy_version("1") is True
        assert is_valid_policy_version("1.0") is True

    def test_invalid_versions(self):
        assert is_valid_policy_version("2") is False
        assert is_valid_policy_version("0") is False
        assert is_valid_policy_version("") is False


class TestDeepMerge:
    """Tests for deep dictionary merging."""

    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"user": {"name": "Alice", "age": 30}}
        override = {"user": {"email": "alice@example.com"}}
        result = deep_merge(base, override)
        assert result == {"user": {"name": "Alice", "age": 30, "email": "alice@example.com"}}

    def test_override_nested(self):
        base = {"user": {"name": "Alice"}}
        override = {"user": {"name": "Bob"}}
        result = deep_merge(base, override)
        assert result == {"user": {"name": "Bob"}}

    def test_original_unchanged(self):
        base = {"a": 1}
        override = {"a": 2}
        result = deep_merge(base, override)
        assert base == {"a": 1}  # Original unchanged
        assert result == {"a": 2}


class TestSecureCompare:
    """Tests for constant-time secure comparison."""

    def test_equal_strings(self):
        assert secure_compare("admin", "admin") is True
        assert secure_compare("user", "user") is True
        assert secure_compare("", "") is True

    def test_unequal_strings(self):
        assert secure_compare("admin", "user") is False
        assert secure_compare("admin", "Admin") is False  # Case-sensitive
        assert secure_compare("admin", "admin ") is False  # Whitespace matters

    def test_unicode_strings(self):
        assert secure_compare("日本語", "日本語") is True
        assert secure_compare("日本語", "中文") is False
        assert secure_compare("émoji", "émoji") is True

    def test_none_values(self):
        # None comparisons should always return False for security
        assert secure_compare(None, None) is False
        assert secure_compare(None, "value") is False
        assert secure_compare("value", None) is False

    def test_integers(self):
        assert secure_compare(123, 123) is True
        assert secure_compare(123, 456) is False
        assert secure_compare(0, 0) is True

    def test_floats(self):
        assert secure_compare(1.5, 1.5) is True
        assert secure_compare(1.5, 1.6) is False

    def test_booleans(self):
        assert secure_compare(True, True) is True
        assert secure_compare(False, False) is True
        assert secure_compare(True, False) is False

    def test_lists(self):
        assert secure_compare([1, 2, 3], [1, 2, 3]) is True
        assert secure_compare([1, 2, 3], [1, 2, 4]) is False
        assert secure_compare([1, 2], [1, 2, 3]) is False
        assert secure_compare(["a", "b"], ["a", "b"]) is True

    def test_mixed_types(self):
        # Different types should return False
        assert secure_compare("123", 123) is False
        assert secure_compare(1, 1.0) is True  # int and float can be equal


class TestSecureContains:
    """Tests for constant-time membership checking."""

    def test_string_in_list(self):
        assert secure_contains("admin", ["user", "admin", "guest"]) is True
        assert secure_contains("root", ["user", "admin", "guest"]) is False

    def test_integer_in_list(self):
        assert secure_contains(2, [1, 2, 3]) is True
        assert secure_contains(4, [1, 2, 3]) is False

    def test_empty_list(self):
        assert secure_contains("admin", []) is False

    def test_not_a_list(self):
        assert secure_contains("admin", "admin") is False
        assert secure_contains("a", {"a": 1}) is False
        assert secure_contains("a", None) is False

    def test_unicode_in_list(self):
        assert secure_contains("日本", ["中国", "日本", "韩国"]) is True
        assert secure_contains("美国", ["中国", "日本", "韩国"]) is False

    def test_case_sensitivity(self):
        assert secure_contains("Admin", ["admin", "user"]) is False
        assert secure_contains("admin", ["Admin", "User"]) is False
