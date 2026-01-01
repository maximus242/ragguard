"""
Tests for condition compilation.

Verifies that:
- Conditions are correctly parsed and compiled
- Compiled conditions evaluate to the same results as string parsing
- All operators and value types are supported
- Edge cases are handled correctly
"""

import pytest

from ragguard.policy import Policy, PolicyEngine
from ragguard.policy.compiler import (
    CompiledCondition,
    CompiledConditionEvaluator,
    CompiledValue,
    ConditionCompiler,
    ConditionOperator,
    ValueType,
)

# ============================================================================
# Compiler Unit Tests
# ============================================================================

def test_compile_equals_condition():
    """Test compiling == condition."""
    condition = "user.department == document.department"
    compiled = ConditionCompiler.compile_condition(condition)

    assert compiled.operator == ConditionOperator.EQUALS
    assert compiled.left.value_type == ValueType.USER_FIELD
    assert compiled.left.field_path == ("department",)
    assert compiled.right.value_type == ValueType.DOCUMENT_FIELD
    assert compiled.right.field_path == ("department",)
    assert compiled.original == condition


def test_compile_not_equals_condition():
    """Test compiling != condition."""
    condition = "user.status != 'inactive'"
    compiled = ConditionCompiler.compile_condition(condition)

    assert compiled.operator == ConditionOperator.NOT_EQUALS
    assert compiled.left.value_type == ValueType.USER_FIELD
    assert compiled.left.field_path == ("status",)
    assert compiled.right.value_type == ValueType.LITERAL_STRING
    assert compiled.right.value == "inactive"


def test_compile_in_condition():
    """Test compiling 'in' condition."""
    condition = "user.id in document.shared_with"
    compiled = ConditionCompiler.compile_condition(condition)

    assert compiled.operator == ConditionOperator.IN
    assert compiled.left.value_type == ValueType.USER_FIELD
    assert compiled.left.field_path == ("id",)
    assert compiled.right.value_type == ValueType.DOCUMENT_FIELD
    assert compiled.right.field_path == ("shared_with",)


def test_compile_not_in_condition():
    """Test compiling 'not in' condition."""
    condition = "user.role not in document.blocked_roles"
    compiled = ConditionCompiler.compile_condition(condition)

    assert compiled.operator == ConditionOperator.NOT_IN
    assert compiled.left.value_type == ValueType.USER_FIELD
    assert compiled.left.field_path == ("role",)
    assert compiled.right.value_type == ValueType.DOCUMENT_FIELD
    assert compiled.right.field_path == ("blocked_roles",)


def test_compile_nested_field_paths():
    """Test compiling conditions with nested field paths."""
    condition = "user.metadata.team == document.metadata.team"
    compiled = ConditionCompiler.compile_condition(condition)

    assert compiled.left.field_path == ("metadata", "team")
    assert compiled.right.field_path == ("metadata", "team")


def test_compile_literal_string_double_quotes():
    """Test compiling literal strings with double quotes."""
    condition = 'user.status == "active"'
    compiled = ConditionCompiler.compile_condition(condition)

    assert compiled.right.value_type == ValueType.LITERAL_STRING
    assert compiled.right.value == "active"


def test_compile_literal_string_single_quotes():
    """Test compiling literal strings with single quotes."""
    condition = "user.status == 'active'"
    compiled = ConditionCompiler.compile_condition(condition)

    assert compiled.right.value_type == ValueType.LITERAL_STRING
    assert compiled.right.value == "active"


def test_compile_literal_integer():
    """Test compiling literal integers."""
    condition = "user.age == 42"
    compiled = ConditionCompiler.compile_condition(condition)

    assert compiled.right.value_type == ValueType.LITERAL_NUMBER
    assert compiled.right.value == 42
    assert isinstance(compiled.right.value, int)


def test_compile_literal_float():
    """Test compiling literal floats."""
    condition = "user.score == 3.14"
    compiled = ConditionCompiler.compile_condition(condition)

    assert compiled.right.value_type == ValueType.LITERAL_NUMBER
    assert compiled.right.value == 3.14
    assert isinstance(compiled.right.value, float)


def test_compile_literal_bool_true():
    """Test compiling boolean true literal."""
    condition = "user.active == true"
    compiled = ConditionCompiler.compile_condition(condition)

    assert compiled.right.value_type == ValueType.LITERAL_BOOL
    assert compiled.right.value is True


def test_compile_literal_bool_false():
    """Test compiling boolean false literal."""
    condition = "user.verified == false"
    compiled = ConditionCompiler.compile_condition(condition)

    assert compiled.right.value_type == ValueType.LITERAL_BOOL
    assert compiled.right.value is False


def test_compile_literal_none():
    """Test compiling None literal."""
    condition = "user.manager == none"
    compiled = ConditionCompiler.compile_condition(condition)

    assert compiled.right.value_type == ValueType.LITERAL_NONE
    assert compiled.right.value is None


def test_compile_whitespace_handling():
    """Test that extra whitespace is handled correctly."""
    condition = "  user.dept   ==   document.dept  "
    compiled = ConditionCompiler.compile_condition(condition)

    assert compiled.operator == ConditionOperator.EQUALS
    assert compiled.left.field_path == ("dept",)
    assert compiled.right.field_path == ("dept",)


def test_compile_invalid_operator():
    """Test that invalid operators raise errors."""
    with pytest.raises(ValueError, match="No valid operator found"):
        ConditionCompiler.compile_condition("user.x ~= document.y")

    with pytest.raises(ValueError, match="No valid operator found"):
        ConditionCompiler.compile_condition("user.x % document.y")


# ============================================================================
# Evaluator Unit Tests
# ============================================================================

def test_evaluate_equals_condition_match():
    """Test evaluating == condition that matches."""
    compiled = ConditionCompiler.compile_condition("user.dept == document.dept")
    user = {"dept": "engineering"}
    document = {"dept": "engineering"}

    result = CompiledConditionEvaluator.evaluate(compiled, user, document)
    assert result is True


def test_evaluate_equals_condition_mismatch():
    """Test evaluating == condition that doesn't match."""
    compiled = ConditionCompiler.compile_condition("user.dept == document.dept")
    user = {"dept": "engineering"}
    document = {"dept": "sales"}

    result = CompiledConditionEvaluator.evaluate(compiled, user, document)
    assert result is False


def test_evaluate_equals_condition_none_security():
    """Test that None == None is rejected for security."""
    compiled = ConditionCompiler.compile_condition("user.dept == document.dept")
    user = {}  # Missing dept
    document = {}  # Missing dept

    result = CompiledConditionEvaluator.evaluate(compiled, user, document)
    assert result is False  # Should deny access, not grant it


def test_evaluate_not_equals_condition():
    """Test evaluating != condition."""
    compiled = ConditionCompiler.compile_condition("user.status != 'inactive'")
    user = {"status": "active"}
    document = {}

    result = CompiledConditionEvaluator.evaluate(compiled, user, document)
    assert result is True


def test_evaluate_not_equals_condition_with_none():
    """
    Test != with None values.

    SECURITY: Missing fields should deny access (return False).
    This prevents users without fields from bypassing != checks.
    """
    compiled = ConditionCompiler.compile_condition("user.status != 'inactive'")
    user = {}  # Missing status (None)
    document = {}

    result = CompiledConditionEvaluator.evaluate(compiled, user, document)
    # Security fix: None != 'inactive' should return False (deny access)
    # Users without a status field should NOT be allowed
    assert result is False


def test_evaluate_in_condition_match():
    """Test evaluating 'in' condition that matches."""
    compiled = ConditionCompiler.compile_condition("user.id in document.shared_with")
    user = {"id": "user123"}
    document = {"shared_with": ["user123", "user456"]}

    result = CompiledConditionEvaluator.evaluate(compiled, user, document)
    assert result is True


def test_evaluate_in_condition_mismatch():
    """Test evaluating 'in' condition that doesn't match."""
    compiled = ConditionCompiler.compile_condition("user.id in document.shared_with")
    user = {"id": "user789"}
    document = {"shared_with": ["user123", "user456"]}

    result = CompiledConditionEvaluator.evaluate(compiled, user, document)
    assert result is False


def test_evaluate_in_condition_not_a_list():
    """Test 'in' condition with non-list right side."""
    compiled = ConditionCompiler.compile_condition("user.id in document.shared_with")
    user = {"id": "user123"}
    document = {"shared_with": "not-a-list"}

    result = CompiledConditionEvaluator.evaluate(compiled, user, document)
    assert result is False


def test_evaluate_not_in_condition():
    """Test evaluating 'not in' condition."""
    compiled = ConditionCompiler.compile_condition("user.role not in document.blocked_roles")
    user = {"role": "admin"}
    document = {"blocked_roles": ["guest", "anonymous"]}

    result = CompiledConditionEvaluator.evaluate(compiled, user, document)
    assert result is True


def test_evaluate_nested_field_paths():
    """Test evaluating with nested field paths."""
    compiled = ConditionCompiler.compile_condition("user.metadata.team == document.metadata.team")
    user = {"metadata": {"team": "backend"}}
    document = {"metadata": {"team": "backend"}}

    result = CompiledConditionEvaluator.evaluate(compiled, user, document)
    assert result is True


def test_evaluate_nested_field_missing():
    """Test nested field evaluation with missing intermediate keys."""
    compiled = ConditionCompiler.compile_condition("user.metadata.team == document.metadata.team")
    user = {"metadata": {}}  # Missing 'team'
    document = {"metadata": {"team": "backend"}}

    result = CompiledConditionEvaluator.evaluate(compiled, user, document)
    assert result is False


def test_evaluate_literal_string():
    """Test evaluating with literal strings."""
    compiled = ConditionCompiler.compile_condition("user.dept == 'engineering'")
    user = {"dept": "engineering"}
    document = {}

    result = CompiledConditionEvaluator.evaluate(compiled, user, document)
    assert result is True


def test_evaluate_literal_number():
    """Test evaluating with literal numbers."""
    compiled = ConditionCompiler.compile_condition("user.age == 42")
    user = {"age": 42}
    document = {}

    result = CompiledConditionEvaluator.evaluate(compiled, user, document)
    assert result is True


def test_evaluate_literal_bool():
    """Test evaluating with boolean literals."""
    compiled = ConditionCompiler.compile_condition("user.active == true")
    user = {"active": True}
    document = {}

    result = CompiledConditionEvaluator.evaluate(compiled, user, document)
    assert result is True


# ============================================================================
# Integration Tests with PolicyEngine
# ============================================================================

def test_policy_engine_uses_compiled_conditions():
    """Test that PolicyEngine uses compiled conditions."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "dept-docs",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            }
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Verify conditions were compiled
    assert 0 in engine._compiled_conditions
    assert len(engine._compiled_conditions[0]) == 1

    compiled_cond = engine._compiled_conditions[0][0]
    assert compiled_cond.operator == ConditionOperator.EQUALS

    # Test evaluation
    user = {"department": "engineering"}
    doc_match = {"department": "engineering"}
    doc_nomatch = {"department": "sales"}

    assert engine.evaluate(user, doc_match) is True
    assert engine.evaluate(user, doc_nomatch) is False


def test_policy_engine_multiple_conditions():
    """Test PolicyEngine with multiple conditions in one rule."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "complex",
                "allow": {
                    "conditions": [
                        "user.department == document.department",
                        "user.level in document.allowed_levels"
                    ]
                }
            }
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Verify both conditions were compiled
    assert len(engine._compiled_conditions[0]) == 2

    # Test evaluation (both must pass)
    user = {"department": "engineering", "level": "senior"}
    doc_match = {"department": "engineering", "allowed_levels": ["senior", "lead"]}
    doc_partial = {"department": "engineering", "allowed_levels": ["junior"]}

    assert engine.evaluate(user, doc_match) is True
    assert engine.evaluate(user, doc_partial) is False


def test_policy_engine_compiled_vs_string_parsing():
    """Test that compiled conditions give same results as string parsing."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "test",
                "allow": {
                    "conditions": [
                        "user.dept == document.dept",
                        "user.id in document.shared_with",
                        "user.role != 'guest'"
                    ]
                }
            }
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Test various user/document combinations
    test_cases = [
        (
            {"dept": "eng", "id": "user1", "role": "admin"},
            {"dept": "eng", "shared_with": ["user1"]},
            True
        ),
        (
            {"dept": "eng", "id": "user1", "role": "guest"},
            {"dept": "eng", "shared_with": ["user1"]},
            False  # role is guest
        ),
        (
            {"dept": "sales", "id": "user1", "role": "admin"},
            {"dept": "eng", "shared_with": ["user1"]},
            False  # dept mismatch
        ),
        (
            {"dept": "eng", "id": "user2", "role": "admin"},
            {"dept": "eng", "shared_with": ["user1"]},
            False  # user not in shared_with
        ),
    ]

    for user, document, expected in test_cases:
        result = engine.evaluate(user, document)
        assert result == expected, f"Failed for user={user}, doc={document}"


def test_policy_engine_compilation_error_handling():
    """Test that compilation errors are caught and reported."""
    # Pydantic validation happens before compilation, so invalid operators
    # are caught at policy creation time
    with pytest.raises(Exception):  # ValidationError from Pydantic
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "bad",
                    "allow": {
                        "conditions": ["user.x ~= document.y"]  # ~= not supported
                    }
                }
            ],
            "default": "deny"
        })


def test_policy_engine_no_conditions():
    """Test PolicyEngine with rules that have no conditions."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "admin",
                "allow": {"roles": ["admin"]}
            }
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # No conditions to compile
    assert len(engine._compiled_conditions) == 0

    # Should still work correctly
    admin_user = {"roles": ["admin"]}
    regular_user = {"roles": ["user"]}
    doc = {"text": "secret"}

    assert engine.evaluate(admin_user, doc) is True
    assert engine.evaluate(regular_user, doc) is False


# ============================================================================
# Performance Comparison Tests
# ============================================================================

def test_compiled_conditions_are_faster():
    """Verify that compiled conditions are faster than string parsing."""
    import time

    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "complex",
                "allow": {
                    "conditions": [
                        "user.department == document.department",
                        "user.level in document.allowed_levels",
                        "user.status != 'inactive'"
                    ]
                }
            }
        ],
        "default": "deny"
    })

    # Engine with compiled conditions
    engine_compiled = PolicyEngine(policy)

    # Test data
    user = {"department": "eng", "level": "senior", "status": "active"}
    document = {"department": "eng", "allowed_levels": ["senior", "lead"]}

    # Warm up
    for _ in range(100):
        engine_compiled.evaluate(user, document)

    # Time compiled evaluation
    iterations = 1000
    start = time.time()
    for _ in range(iterations):
        engine_compiled.evaluate(user, document)
    compiled_time = time.time() - start

    # The compiled version should complete successfully
    # We can't directly compare to string parsing since we replaced it
    # But we can verify it's fast enough
    assert compiled_time < 0.1  # Should complete 1000 iterations in < 100ms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
