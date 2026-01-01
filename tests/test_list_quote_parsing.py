"""
Test list quote parsing edge cases.

Verifies that the quote-aware parser correctly handles escaped quotes
and other edge cases in list literals.
"""

import pytest

from ragguard.policy.compiler import ConditionCompiler


def test_list_with_normal_strings():
    """Test that normal string lists work correctly."""
    condition = "user.role in ['admin', 'user', 'guest']"
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None


def test_list_with_double_quoted_strings():
    """Test lists with double-quoted strings."""
    condition = 'user.role in ["admin", "user", "guest"]'
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None


def test_list_with_mixed_quotes():
    """Test lists with mixed quote types."""
    condition = """user.role in ['admin', "user", 'guest']"""
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None


def test_list_with_escaped_quotes():
    """Test that lists with escaped quotes are handled correctly."""
    # String containing escaped single quote
    condition = r"user.message in ['He said \"hello\"', 'other']"
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None


def test_list_with_escaped_quotes_in_double_quotes():
    """Test escaped quotes within double-quoted strings."""
    condition = r'user.message in ["He said \"hello\"", "other"]'
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None


def test_list_unclosed_single_quote():
    """Test that unclosed single quotes are detected."""
    condition = "user.role in ['admin', 'user, 'guest']"  # Missing close quote on 'user
    with pytest.raises(ValueError, match="unclosed single quote"):
        ConditionCompiler.compile_expression(condition)


def test_list_unclosed_double_quote():
    """Test that unclosed double quotes are detected."""
    condition = 'user.role in ["admin", "user, "guest"]'  # Missing close quote on "user
    with pytest.raises(ValueError, match="unclosed double quote"):
        ConditionCompiler.compile_expression(condition)


def test_list_extra_opening_quote():
    """Test that extra opening quotes are detected."""
    condition = "user.role in ['admin', ''user', 'guest']"  # Extra opening quote
    with pytest.raises(ValueError, match="unclosed single quote"):
        ConditionCompiler.compile_expression(condition)


def test_list_with_empty_string():
    """Test that empty strings in lists work correctly."""
    condition = "user.role in ['admin', '', 'guest']"
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None


def test_list_with_comma_in_string():
    """Test that commas inside quoted strings don't split the list incorrectly."""
    condition = r"user.tag in ['item1', 'a,b,c', 'item3']"
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None
    # The list should have 3 items, not 5


def test_list_with_nested_brackets():
    """
    Test that nested brackets in strings are detected (known limitation).

    Currently, brackets inside quoted strings are detected as nested lists.
    This is a known limitation of the simple bracket check.
    """
    condition = r"user.data in ['[1,2,3]', 'normal']"
    # This currently raises an error due to brackets inside the string
    with pytest.raises(ValueError, match="Nested lists are not supported"):
        ConditionCompiler.compile_expression(condition)


def test_list_with_backslash_escape():
    """Test that backslash escapes work correctly."""
    # Escaped backslash before quote
    condition = r"user.path in ['C:\\Users\\test', 'other']"
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None


def test_list_with_unicode():
    """Test that Unicode strings work in lists."""
    condition = "user.name in ['Alice', 'Bo弱', '测试']"
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None


def test_list_with_numbers():
    """Test that numeric values work in lists."""
    condition = "user.id in [1, 2, 3, 42]"
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None


def test_list_with_mixed_types():
    """Test that lists with mixed types work."""
    condition = "user.value in [1, 'two', 3.14, true]"
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None


def test_list_with_whitespace():
    """Test that whitespace in lists is handled correctly."""
    condition = "user.role in [  'admin'  ,  'user'  ,  'guest'  ]"
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None


def test_empty_list():
    """Test that empty lists work."""
    condition = "user.role in []"
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None


def test_list_single_item():
    """Test that single-item lists work."""
    condition = "user.role in ['admin']"
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None


def test_list_with_special_characters():
    """Test lists with special characters in strings."""
    condition = r"user.special in ['@#$%', '!&*()', '+={}']"
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None


def test_list_in_complex_expression():
    """Test that lists work correctly in complex expressions."""
    condition = "user.role in ['admin', 'mod'] AND user.active == true"
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None


def test_list_mismatched_brackets():
    """Test that mismatched brackets are detected."""
    condition = "user.role in ['admin', 'user'"  # Missing closing ]
    with pytest.raises(ValueError, match="missing closing bracket"):
        ConditionCompiler.compile_expression(condition)


def test_quote_state_tracking():
    """
    Test that quote state is tracked correctly across the entire list.

    This is the key test for the quote parsing fix - ensures that
    we properly track whether we're inside quotes or not.
    """
    # This should succeed - quotes are balanced
    condition = r"user.msg in ['He said \"hi\"', 'She said \"bye\"']"
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None

    # This should fail - unclosed quote
    bad_condition = r"user.msg in ['He said \"hi\", 'She said \"bye\"']"  # Missing close quote after hi"
    with pytest.raises(ValueError, match="unclosed"):
        ConditionCompiler.compile_expression(bad_condition)
