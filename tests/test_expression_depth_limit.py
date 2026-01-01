"""
Test expression depth limit enforcement.

Verifies that the off-by-one bug in depth checking is fixed.
The limit should reject expressions at exactly MAX_EXPRESSION_DEPTH,
not one level higher.

NOTE: Depth is counted by AND/OR nesting levels, not parentheses.
"""

import pytest

from ragguard.policy.compiler import ConditionCompiler
from ragguard.policy.models import PolicyLimits


def _build_nested_expression(depth: int, counter: int = 0) -> str:
    """
    Build a nested OR expression with the given depth.

    The depth is counted during recursive parsing of OR operations.
    Each level of OR nesting increases the parsing depth by 1.

    depth=0: "user.x0 == 0"
    depth=1: "user.x1 == 1 OR user.y1 == 1"
    depth=2: "(user.x2 == 2 OR user.y2 == 2) OR user.z2 == 2"
    etc.
    """
    if depth == 0:
        return f"user.x{counter} == {counter}"

    # Create nested OR structure
    # Left side: nested expression at depth - 1
    # Right side: simple condition
    left = _build_nested_expression(depth - 1, counter + 1)
    right = f"user.z{counter} == {counter}"
    return f"({left}) OR {right}"


def test_expression_depth_at_limit():
    """
    Test that expressions at exactly MAX_EXPRESSION_DEPTH are rejected.

    This verifies the fix for the off-by-one bug where depth > limit
    was changed to depth >= limit.
    """
    max_depth = PolicyLimits.MAX_EXPRESSION_DEPTH

    # Build a deeply nested AND expression at exactly the limit
    condition = _build_nested_expression(max_depth)

    # Should raise ValueError for exceeding depth limit
    with pytest.raises(ValueError, match="Expression nesting too deep"):
        ConditionCompiler.compile_expression(condition)


def test_expression_depth_one_below_limit():
    """
    Test that expressions one level below the limit are accepted.
    """
    max_depth = PolicyLimits.MAX_EXPRESSION_DEPTH

    # Build expression at max_depth - 1
    condition = _build_nested_expression(max_depth - 1)

    # Should compile successfully
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None


def test_expression_depth_simple_condition():
    """
    Test that simple conditions (depth 0) work correctly.
    """
    condition = "user.role == 'admin'"
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None


def test_expression_depth_and_or_chains():
    """
    Test that AND/OR chains don't incorrectly trigger depth limits.

    Flat chains like (A AND B AND C) should count as depth 1, not depth 3.
    """
    # Long chain of AND conditions (flat, depth = 1)
    condition = " AND ".join([f"user.field{i} == {i}" for i in range(10)])
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None

    # Long chain of OR conditions (flat, depth = 1)
    condition = " OR ".join([f"user.field{i} == {i}" for i in range(10)])
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None


def test_expression_depth_mixed_nesting():
    """
    Test that mixed nesting of AND/OR is counted correctly.
    """
    # Nested structure: ((A AND B) OR (C AND D)) = depth 2
    condition = "((user.a == 1 AND user.b == 2) OR (user.c == 3 AND user.d == 4))"
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None


def test_expression_depth_exceeds_limit_by_one():
    """
    Test that expressions exceeding the limit by 1 level are rejected.

    This is the critical test - with the bug, this would have been accepted.
    """
    max_depth = PolicyLimits.MAX_EXPRESSION_DEPTH

    # Build expression at max_depth + 1 (one level too deep)
    condition = _build_nested_expression(max_depth + 1)

    # Should raise ValueError
    with pytest.raises(ValueError, match="Expression nesting too deep"):
        ConditionCompiler.compile_expression(condition)


def test_expression_depth_far_exceeds_limit():
    """
    Test that expressions far exceeding the limit are rejected.
    """
    max_depth = PolicyLimits.MAX_EXPRESSION_DEPTH

    # Build expression at 2x the limit
    condition = _build_nested_expression(max_depth * 2)

    # Should raise ValueError
    with pytest.raises(ValueError, match="Expression nesting too deep"):
        ConditionCompiler.compile_expression(condition)


def test_expression_depth_limit_value():
    """
    Test that the depth limit constant is reasonable.

    The limit should be high enough for complex policies but
    low enough to prevent DoS attacks.
    """
    assert PolicyLimits.MAX_EXPRESSION_DEPTH >= 10  # At least 10 for complex policies
    assert PolicyLimits.MAX_EXPRESSION_DEPTH <= 100  # At most 100 to prevent DoS


def test_expression_depth_error_message():
    """
    Test that the error message is helpful and includes depth information.
    """
    max_depth = PolicyLimits.MAX_EXPRESSION_DEPTH
    condition = _build_nested_expression(max_depth + 5)

    with pytest.raises(ValueError) as exc_info:
        ConditionCompiler.compile_expression(condition)

    error_msg = str(exc_info.value)
    assert "Expression nesting too deep" in error_msg
    assert str(max_depth) in error_msg  # Should mention the limit


def test_expression_depth_with_list_literals():
    """
    Test that list literals don't incorrectly affect depth counting.
    """
    # List literal shouldn't count toward expression depth
    condition = "user.role in ['admin', 'moderator', 'user']"
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None

    # List literal with OR creates depth 1
    condition = "user.role in ['admin', 'moderator'] OR user.active == true"
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None


def test_expression_depth_boundary_cases():
    """
    Test various boundary cases around the depth limit.
    """
    max_depth = PolicyLimits.MAX_EXPRESSION_DEPTH

    # Test at limit - 2, limit - 1, limit, limit + 1, limit + 2
    test_cases = [
        (max_depth - 2, True, "Two below limit"),
        (max_depth - 1, True, "One below limit"),
        (max_depth, False, "At limit (should fail)"),
        (max_depth + 1, False, "One above limit"),
        (max_depth + 2, False, "Two above limit"),
    ]

    for depth, should_succeed, description in test_cases:
        if depth < 0:
            continue  # Skip invalid depths

        condition = _build_nested_expression(depth)

        if should_succeed:
            # Should compile successfully
            try:
                compiled = ConditionCompiler.compile_expression(condition)
                assert compiled is not None, f"Failed: {description}"
            except ValueError:
                pytest.fail(f"Unexpectedly failed at depth {depth}: {description}")
        else:
            # Should raise ValueError
            with pytest.raises(ValueError, match="Expression nesting too deep"):
                ConditionCompiler.compile_expression(condition)
