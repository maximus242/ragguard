#!/usr/bin/env python3
"""
Comprehensive tests for OR logic support in RAGGuard v0.3.0.

Tests cover:
- Basic OR expressions
- OR with AND combinations
- Nested expressions
- Edge cases and error handling
- Backward compatibility with v0.2.0 policies
"""

import pytest

from ragguard import Policy
from ragguard.policy.compiler import (
    CompiledCondition,
    CompiledConditionEvaluator,
    CompiledExpression,
    ConditionCompiler,
    LogicalOperator,
)

# ============================================================================
# Compiler Tests - OR Logic Parsing
# ============================================================================

def test_simple_or_expression():
    """Test parsing a simple OR expression."""
    condition = "(user.role == 'admin' OR user.role == 'manager')"
    compiled = ConditionCompiler.compile_expression(condition)

    assert isinstance(compiled, CompiledExpression)
    assert compiled.operator == LogicalOperator.OR
    assert len(compiled.children) == 2
    assert all(isinstance(child, CompiledCondition) for child in compiled.children)


def test_simple_and_expression():
    """Test parsing a simple AND expression."""
    condition = "user.department == 'engineering' AND user.level >= 5"
    compiled = ConditionCompiler.compile_expression(condition)

    assert isinstance(compiled, CompiledExpression)
    assert compiled.operator == LogicalOperator.AND
    assert len(compiled.children) == 2


def test_or_with_three_conditions():
    """Test OR with three conditions."""
    condition = "user.role == 'admin' OR user.role == 'manager' OR user.role == 'owner'"
    compiled = ConditionCompiler.compile_expression(condition)

    assert isinstance(compiled, CompiledExpression)
    assert compiled.operator == LogicalOperator.OR
    assert len(compiled.children) == 3


def test_nested_or_and():
    """Test nested OR and AND expressions."""
    condition = "(user.role == 'admin' OR user.role == 'manager') AND document.status == 'published'"
    compiled = ConditionCompiler.compile_expression(condition)

    assert isinstance(compiled, CompiledExpression)
    assert compiled.operator == LogicalOperator.AND
    assert len(compiled.children) == 2

    # First child should be the OR expression
    or_expr = compiled.children[0]
    assert isinstance(or_expr, CompiledExpression)
    assert or_expr.operator == LogicalOperator.OR
    assert len(or_expr.children) == 2

    # Second child should be a simple condition
    assert isinstance(compiled.children[1], CompiledCondition)


def test_deeply_nested_expression():
    """Test deeply nested expression."""
    condition = "((user.role == 'admin' OR user.role == 'manager') AND document.status == 'published') OR user.id == document.owner_id"
    compiled = ConditionCompiler.compile_expression(condition)

    assert isinstance(compiled, CompiledExpression)
    assert compiled.operator == LogicalOperator.OR
    assert len(compiled.children) == 2


def test_backward_compatibility_simple_condition():
    """Test that simple conditions without OR/AND still work."""
    condition = "user.department == document.department"
    compiled = ConditionCompiler.compile_expression(condition)

    # Should return a CompiledCondition, not CompiledExpression
    assert isinstance(compiled, CompiledCondition)
    assert not isinstance(compiled, CompiledExpression)


def test_parentheses_unwrapping():
    """Test that outer parentheses are properly unwrapped."""
    condition1 = "(user.role == 'admin')"
    condition2 = "user.role == 'admin'"

    compiled1 = ConditionCompiler.compile_expression(condition1)
    compiled2 = ConditionCompiler.compile_expression(condition2)

    # Both should result in simple CompiledCondition (parentheses unwrapped)
    assert isinstance(compiled1, CompiledCondition)
    assert isinstance(compiled2, CompiledCondition)


def test_or_precedence():
    """Test that OR has lower precedence than AND."""
    # A AND B OR C should be (A AND B) OR C
    condition = "user.dept == 'eng' AND user.level >= 5 OR user.role == 'admin'"
    compiled = ConditionCompiler.compile_expression(condition)

    assert isinstance(compiled, CompiledExpression)
    assert compiled.operator == LogicalOperator.OR
    assert len(compiled.children) == 2

    # First child should be AND expression
    and_expr = compiled.children[0]
    assert isinstance(and_expr, CompiledExpression)
    assert and_expr.operator == LogicalOperator.AND


def test_case_insensitive_operators():
    """Test that OR/AND are case-insensitive."""
    condition1 = "user.role == 'admin' OR user.role == 'manager'"
    condition2 = "user.role == 'admin' or user.role == 'manager'"
    condition3 = "user.role == 'admin' Or user.role == 'manager'"

    compiled1 = ConditionCompiler.compile_expression(condition1)
    compiled2 = ConditionCompiler.compile_expression(condition2)
    compiled3 = ConditionCompiler.compile_expression(condition3)

    assert all(isinstance(c, CompiledExpression) for c in [compiled1, compiled2, compiled3])
    assert all(c.operator == LogicalOperator.OR for c in [compiled1, compiled2, compiled3])


# ============================================================================
# Compiler Tests - Error Handling
# ============================================================================

def test_unbalanced_parentheses_open():
    """Test error on unbalanced parentheses (missing close)."""
    condition = "(user.role == 'admin' OR user.role == 'manager'"

    with pytest.raises(ValueError, match="Unbalanced parentheses"):
        ConditionCompiler.compile_expression(condition)


def test_unbalanced_parentheses_close():
    """Test error on unbalanced parentheses (missing open)."""
    condition = "user.role == 'admin' OR user.role == 'manager')"

    with pytest.raises(ValueError, match="Unbalanced parentheses"):
        ConditionCompiler.compile_expression(condition)


def test_or_in_quoted_string_ignored():
    """Test that OR inside quoted strings doesn't trigger expression parsing."""
    condition = "document.title == 'admin OR manager'"
    compiled = ConditionCompiler.compile_expression(condition)

    # Should be a simple condition, not an expression
    assert isinstance(compiled, CompiledCondition)


def test_and_in_quoted_string_ignored():
    """Test that AND inside quoted strings doesn't trigger expression parsing."""
    condition = "document.title == 'engineering AND sales'"
    compiled = ConditionCompiler.compile_expression(condition)

    # Should be a simple condition, not an expression
    assert isinstance(compiled, CompiledCondition)


# ============================================================================
# Evaluator Tests - OR Logic Evaluation
# ============================================================================

def test_evaluate_simple_or_true_first():
    """Test OR evaluation when first condition is true."""
    condition = "user.role == 'admin' OR user.role == 'manager'"
    compiled = ConditionCompiler.compile_expression(condition)

    user = {"role": "admin"}
    document = {}

    result = CompiledConditionEvaluator.evaluate_node(compiled, user, document)
    assert result is True


def test_evaluate_simple_or_true_second():
    """Test OR evaluation when second condition is true."""
    condition = "user.role == 'admin' OR user.role == 'manager'"
    compiled = ConditionCompiler.compile_expression(condition)

    user = {"role": "manager"}
    document = {}

    result = CompiledConditionEvaluator.evaluate_node(compiled, user, document)
    assert result is True


def test_evaluate_simple_or_both_true():
    """Test OR evaluation when both conditions are true."""
    condition = "user.active == true OR user.verified == true"
    compiled = ConditionCompiler.compile_expression(condition)

    user = {"active": True, "verified": True}
    document = {}

    result = CompiledConditionEvaluator.evaluate_node(compiled, user, document)
    assert result is True


def test_evaluate_simple_or_both_false():
    """Test OR evaluation when both conditions are false."""
    condition = "user.role == 'admin' OR user.role == 'manager'"
    compiled = ConditionCompiler.compile_expression(condition)

    user = {"role": "guest"}
    document = {}

    result = CompiledConditionEvaluator.evaluate_node(compiled, user, document)
    assert result is False


def test_evaluate_simple_and_both_true():
    """Test AND evaluation when both conditions are true."""
    condition = "user.department == 'engineering' AND user.level >= 5"
    compiled = ConditionCompiler.compile_expression(condition)

    user = {"department": "engineering", "level": 7}
    document = {}

    result = CompiledConditionEvaluator.evaluate_node(compiled, user, document)
    assert result is True


def test_evaluate_simple_and_first_false():
    """Test AND evaluation when first condition is false."""
    condition = "user.department == 'engineering' AND user.level >= 5"
    compiled = ConditionCompiler.compile_expression(condition)

    user = {"department": "sales", "level": 7}
    document = {}

    result = CompiledConditionEvaluator.evaluate_node(compiled, user, document)
    assert result is False


def test_evaluate_nested_or_and():
    """Test nested OR and AND evaluation."""
    condition = "(user.role == 'admin' OR user.role == 'manager') AND document.status == 'published'"
    compiled = ConditionCompiler.compile_expression(condition)

    # Test 1: Admin with published document - should pass
    user1 = {"role": "admin"}
    document1 = {"status": "published"}
    assert CompiledConditionEvaluator.evaluate_node(compiled, user1, document1) is True

    # Test 2: Manager with published document - should pass
    user2 = {"role": "manager"}
    document2 = {"status": "published"}
    assert CompiledConditionEvaluator.evaluate_node(compiled, user2, document2) is True

    # Test 3: Admin with draft document - should fail
    user3 = {"role": "admin"}
    document3 = {"status": "draft"}
    assert CompiledConditionEvaluator.evaluate_node(compiled, user3, document3) is False

    # Test 4: Guest with published document - should fail
    user4 = {"role": "guest"}
    document4 = {"status": "published"}
    assert CompiledConditionEvaluator.evaluate_node(compiled, user4, document4) is False


def test_evaluate_complex_nested():
    """Test complex nested expression evaluation."""
    condition = "((user.role == 'admin' OR user.role == 'manager') AND document.status == 'published') OR user.id == document.owner_id"
    compiled = ConditionCompiler.compile_expression(condition)

    # Test 1: Admin with published doc - should pass via first branch
    user1 = {"role": "admin", "id": "user123"}
    document1 = {"status": "published", "owner_id": "user456"}
    assert CompiledConditionEvaluator.evaluate_node(compiled, user1, document1) is True

    # Test 2: Guest who owns the doc - should pass via second branch
    user2 = {"role": "guest", "id": "user123"}
    document2 = {"status": "draft", "owner_id": "user123"}
    assert CompiledConditionEvaluator.evaluate_node(compiled, user2, document2) is True

    # Test 3: Guest who doesn't own draft doc - should fail
    user3 = {"role": "guest", "id": "user123"}
    document3 = {"status": "draft", "owner_id": "user456"}
    assert CompiledConditionEvaluator.evaluate_node(compiled, user3, document3) is False


def test_evaluate_three_way_or():
    """Test OR with three conditions."""
    condition = "user.role == 'admin' OR user.role == 'manager' OR user.role == 'owner'"
    compiled = ConditionCompiler.compile_expression(condition)

    # Test each role
    for role in ["admin", "manager", "owner"]:
        user = {"role": role}
        document = {}
        assert CompiledConditionEvaluator.evaluate_node(compiled, user, document) is True

    # Test non-matching role
    user = {"role": "guest"}
    document = {}
    assert CompiledConditionEvaluator.evaluate_node(compiled, user, document) is False


# ============================================================================
# Policy Engine Integration Tests
# ============================================================================

def test_policy_with_or_logic():
    """Test full policy evaluation with OR logic."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "senior-access",
                "allow": {
                    "conditions": [
                        "(user.role == 'admin' OR user.role == 'manager')",
                        "document.status == 'published'"
                    ]
                }
            }
        ],
        "default": "deny"
    })

    # Test 1: Admin with published document - should pass
    from ragguard.policy.engine import PolicyEngine
    engine = PolicyEngine(policy)

    user1 = {"role": "admin"}
    document1 = {"status": "published"}
    assert engine.evaluate(user1, document1) is True

    # Test 2: Manager with published document - should pass
    user2 = {"role": "manager"}
    document2 = {"status": "published"}
    assert engine.evaluate(user2, document2) is True

    # Test 3: Guest with published document - should fail (first condition fails)
    user3 = {"role": "guest"}
    document3 = {"status": "published"}
    assert engine.evaluate(user3, document3) is False

    # Test 4: Admin with draft document - should fail (second condition fails)
    user4 = {"role": "admin"}
    document4 = {"status": "draft"}
    assert engine.evaluate(user4, document4) is False


def test_policy_multiple_or_rules():
    """Test policy with multiple rules using OR logic."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "senior-or-owner",
                "allow": {
                    "conditions": [
                        "user.role == 'admin' OR user.role == 'manager' OR user.id == document.owner_id"
                    ]
                }
            }
        ],
        "default": "deny"
    })

    from ragguard.policy.engine import PolicyEngine
    engine = PolicyEngine(policy)

    # Test 1: Admin - should pass
    user1 = {"role": "admin", "id": "user123"}
    document1 = {"owner_id": "user456"}
    assert engine.evaluate(user1, document1) is True

    # Test 2: Document owner - should pass
    user2 = {"role": "guest", "id": "user123"}
    document2 = {"owner_id": "user123"}
    assert engine.evaluate(user2, document2) is True

    # Test 3: Guest who doesn't own doc - should fail
    user3 = {"role": "guest", "id": "user123"}
    document3 = {"owner_id": "user456"}
    assert engine.evaluate(user3, document3) is False


def test_backward_compatibility_no_or():
    """Test that v0.2.0 policies without OR logic still work."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "dept-match",
                "allow": {
                    "conditions": [
                        "user.department == document.department"
                    ]
                }
            }
        ],
        "default": "deny"
    })

    from ragguard.policy.engine import PolicyEngine
    engine = PolicyEngine(policy)

    # Test 1: Matching department - should pass
    user1 = {"department": "engineering"}
    document1 = {"department": "engineering"}
    assert engine.evaluate(user1, document1) is True

    # Test 2: Different departments - should fail
    user2 = {"department": "engineering"}
    document2 = {"department": "sales"}
    assert engine.evaluate(user2, document2) is False


def test_or_with_array_operations():
    """Test OR logic combined with array operations."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "shared-or-public",
                "allow": {
                    "conditions": [
                        "user.id in document.shared_with OR document.visibility == 'public'"
                    ]
                }
            }
        ],
        "default": "deny"
    })

    from ragguard.policy.engine import PolicyEngine
    engine = PolicyEngine(policy)

    # Test 1: User in shared_with list - should pass
    user1 = {"id": "alice"}
    document1 = {"shared_with": ["alice", "bob"], "visibility": "private"}
    assert engine.evaluate(user1, document1) is True

    # Test 2: Public document - should pass even if not in shared_with
    user2 = {"id": "charlie"}
    document2 = {"shared_with": ["alice", "bob"], "visibility": "public"}
    assert engine.evaluate(user2, document2) is True

    # Test 3: Private doc, user not in shared_with - should fail
    user3 = {"id": "charlie"}
    document3 = {"shared_with": ["alice", "bob"], "visibility": "private"}
    assert engine.evaluate(user3, document3) is False


def test_or_with_field_existence():
    """Test OR logic combined with field existence checks."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "admin-or-no-sensitivity",
                "allow": {
                    "conditions": [
                        "user.role == 'admin' OR document.sensitivity not exists"
                    ]
                }
            }
        ],
        "default": "deny"
    })

    from ragguard.policy.engine import PolicyEngine
    engine = PolicyEngine(policy)

    # Test 1: Admin - should pass regardless of sensitivity
    user1 = {"role": "admin"}
    document1 = {"sensitivity": "high"}
    assert engine.evaluate(user1, document1) is True

    # Test 2: Non-admin with no sensitivity field - should pass
    user2 = {"role": "user"}
    document2 = {"title": "Public doc"}
    assert engine.evaluate(user2, document2) is True

    # Test 3: Non-admin with sensitivity field - should fail
    user3 = {"role": "user"}
    document3 = {"sensitivity": "high"}
    assert engine.evaluate(user3, document3) is False


def test_or_with_comparison_operators():
    """Test OR logic combined with comparison operators."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "high-level-or-owner",
                "allow": {
                    "conditions": [
                        "user.level >= 8 OR user.id == document.owner_id"
                    ]
                }
            }
        ],
        "default": "deny"
    })

    from ragguard.policy.engine import PolicyEngine
    engine = PolicyEngine(policy)

    # Test 1: High level user - should pass
    user1 = {"id": "alice", "level": 9}
    document1 = {"owner_id": "bob"}
    assert engine.evaluate(user1, document1) is True

    # Test 2: Document owner with low level - should pass
    user2 = {"id": "alice", "level": 3}
    document2 = {"owner_id": "alice"}
    assert engine.evaluate(user2, document2) is True

    # Test 3: Low level non-owner - should fail
    user3 = {"id": "alice", "level": 3}
    document3 = {"owner_id": "bob"}
    assert engine.evaluate(user3, document3) is False


# ============================================================================
# Performance Tests
# ============================================================================

import os


@pytest.mark.benchmark
@pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("SKIP_PERF_TESTS") == "1",
    reason="Skipping performance tests in CI"
)
def test_or_logic_performance():
    """Test that OR logic doesn't significantly impact performance."""
    import time

    # Create policy with OR logic
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "multi-role",
                "allow": {
                    "conditions": [
                        "user.role == 'admin' OR user.role == 'manager' OR user.role == 'owner' OR user.role == 'lead'"
                    ]
                }
            }
        ],
        "default": "deny"
    })

    from ragguard.policy.engine import PolicyEngine
    engine = PolicyEngine(policy)

    user = {"role": "manager"}
    document = {}

    # Warm up
    for _ in range(1000):
        engine.evaluate(user, document)

    # Measure
    start = time.perf_counter()
    iterations = 10000
    for _ in range(iterations):
        engine.evaluate(user, document)
    end = time.perf_counter()

    avg_time_us = (end - start) / iterations * 1_000_000

    # Should be under 50μs per evaluation (generous threshold for system variability)
    assert avg_time_us < 50, f"OR logic evaluation too slow: {avg_time_us:.2f}μs (expected <50μs)"

    print(f"✅ OR logic performance: {avg_time_us:.2f}μs per evaluation")


if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v"])
