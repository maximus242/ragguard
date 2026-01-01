#!/usr/bin/env python3
"""
Tests for OR logic complexity limits (DoS prevention).

Tests that excessive OR/AND expressions are rejected to prevent:
- Stack overflow attacks
- Exponential time complexity
- Resource exhaustion
"""

import pytest

from ragguard.policy.compiler import ConditionCompiler
from ragguard.policy.models import PolicyLimits


def test_expression_depth_limit():
    """Test that deeply nested expressions are rejected."""
    # Build expression that exceeds MAX_EXPRESSION_DEPTH
    condition = "user.role == 'admin'"
    for i in range(PolicyLimits.MAX_EXPRESSION_DEPTH + 2):
        condition = f"({condition} OR user.role == 'user{i}')"

    with pytest.raises(ValueError, match="Expression nesting too deep"):
        ConditionCompiler.compile_expression(condition)


def test_or_branch_count_limit():
    """Test that expressions with too many OR branches are rejected."""
    # Build expression with MAX_EXPRESSION_BRANCHES + 1 branches
    branches = []
    for i in range(PolicyLimits.MAX_EXPRESSION_BRANCHES + 1):
        branches.append(f"user.role == 'role{i}'")

    condition = " OR ".join(branches)

    with pytest.raises(ValueError, match="Too many OR branches"):
        ConditionCompiler.compile_expression(condition)


def test_and_branch_count_limit():
    """Test that expressions with too many AND branches are rejected."""
    # Build expression with MAX_EXPRESSION_BRANCHES + 1 branches
    branches = []
    for i in range(PolicyLimits.MAX_EXPRESSION_BRANCHES + 1):
        branches.append(f"user.field{i} == 'value{i}'")

    condition = " AND ".join(branches)

    with pytest.raises(ValueError, match="Too many AND branches"):
        ConditionCompiler.compile_expression(condition)


def test_total_conditions_limit():
    """Test that expressions with too many total conditions are rejected."""
    # Build a tree with many conditions using nested expressions
    # We need to stay within branch limits but exceed total conditions
    # Use: (A AND B AND C...) OR (D AND E AND F...) with small branches

    # Build groups of AND conditions (won't trigger branch limit)
    num_groups = 10
    conditions_per_group = 15  # Total: 150 conditions > 100 limit

    groups = []
    for g in range(num_groups):
        and_conditions = []
        for i in range(conditions_per_group):
            and_conditions.append(f"user.field{g}_{i} == 'value{i}'")
        groups.append(f"({' AND '.join(and_conditions)})")

    condition = " OR ".join(groups)

    with pytest.raises(ValueError, match="Too many conditions in expression"):
        ConditionCompiler.compile_expression(condition)


def test_within_depth_limit():
    """Test that expressions within depth limit are accepted."""
    # Build expression at exactly MAX_EXPRESSION_DEPTH
    condition = "user.role == 'admin'"
    for i in range(PolicyLimits.MAX_EXPRESSION_DEPTH - 1):
        condition = f"({condition} OR user.role == 'user{i}')"

    # Should not raise
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None


def test_within_branch_limit():
    """Test that expressions within branch limit are accepted."""
    # Build expression with exactly MAX_EXPRESSION_BRANCHES
    branches = []
    for i in range(PolicyLimits.MAX_EXPRESSION_BRANCHES):
        branches.append(f"user.role == 'role{i}'")

    condition = " OR ".join(branches)

    # Should not raise
    compiled = ConditionCompiler.compile_expression(condition)
    assert compiled is not None


def test_within_conditions_limit():
    """Test that expressions within conditions limit are accepted."""
    # Build expression with exactly MAX_EXPRESSION_CONDITIONS
    branches = []
    for i in range(PolicyLimits.MAX_EXPRESSION_CONDITIONS):
        branches.append(f"user.field{i} == 'value{i}'")

    condition = " OR ".join(branches)

    # Should not raise (though may hit branch limit first)
    # Use AND instead to test conditions limit specifically
    if PolicyLimits.MAX_EXPRESSION_CONDITIONS <= PolicyLimits.MAX_EXPRESSION_BRANCHES:
        condition = " AND ".join(branches)
        compiled = ConditionCompiler.compile_expression(condition)
        assert compiled is not None


def test_complexity_error_messages():
    """Test that complexity errors have helpful messages."""
    # Test depth limit error
    condition = "user.role == 'admin'"
    for i in range(PolicyLimits.MAX_EXPRESSION_DEPTH + 2):
        condition = f"({condition} OR user.role == 'user{i}')"

    try:
        ConditionCompiler.compile_expression(condition)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        error_msg = str(e)
        assert "Expression nesting too deep" in error_msg
        assert "Try simplifying" in error_msg
        assert str(PolicyLimits.MAX_EXPRESSION_DEPTH) in error_msg


def test_count_conditions_simple():
    """Test count_conditions() method for simple expression."""
    condition = "user.role == 'admin' OR user.role == 'manager'"
    compiled = ConditionCompiler.compile_expression(condition)

    assert compiled.count_conditions() == 2


def test_count_conditions_nested():
    """Test count_conditions() method for nested expression."""
    condition = "(user.role == 'admin' OR user.role == 'manager') AND document.status == 'published'"
    compiled = ConditionCompiler.compile_expression(condition)

    assert compiled.count_conditions() == 3


def test_count_conditions_deep():
    """Test count_conditions() method for deeply nested expression."""
    condition = "((user.role == 'admin' OR user.role == 'manager') AND document.status == 'published') OR user.id == document.owner_id"
    compiled = ConditionCompiler.compile_expression(condition)

    assert compiled.count_conditions() == 4


def test_get_depth_simple():
    """Test get_depth() method for simple expression."""
    condition = "user.role == 'admin' OR user.role == 'manager'"
    compiled = ConditionCompiler.compile_expression(condition)

    assert compiled.get_depth() == 1


def test_get_depth_nested():
    """Test get_depth() method for nested expression."""
    condition = "(user.role == 'admin' OR user.role == 'manager') AND document.status == 'published'"
    compiled = ConditionCompiler.compile_expression(condition)

    assert compiled.get_depth() == 2


def test_get_depth_deeply_nested():
    """Test get_depth() method for deeply nested expression."""
    condition = "((user.role == 'admin' OR user.role == 'manager') AND document.status == 'published') OR user.id == document.owner_id"
    compiled = ConditionCompiler.compile_expression(condition)

    assert compiled.get_depth() == 3


def test_policy_with_complex_expression():
    """Test that PolicyEngine validates complexity limits."""
    from ragguard import Policy
    from ragguard.policy.engine import PolicyEngine

    # Build policy with expression that exceeds limits
    branches = []
    for i in range(PolicyLimits.MAX_EXPRESSION_BRANCHES + 1):
        branches.append(f"user.role == 'role{i}'")

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "complex",
            "allow": {
                "conditions": [" OR ".join(branches)]
            }
        }],
        "default": "deny"
    })

    # Error occurs during PolicyEngine initialization (compilation)
    with pytest.raises(Exception, match="Too many OR branches"):
        PolicyEngine(policy)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
