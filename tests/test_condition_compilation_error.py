"""
Tests for ConditionCompilationError exception.
"""

import pytest

from ragguard.exceptions import ConditionCompilationError, PolicyError
from ragguard.policy import Policy
from ragguard.policy.engine import PolicyEngine


class TestConditionCompilationError:
    """Test the ConditionCompilationError exception."""

    def test_basic_message(self):
        """Test basic error message."""
        error = ConditionCompilationError(
            condition="user.department ==",
            rule_name="dept-access"
        )
        assert "user.department ==" in str(error)
        assert "dept-access" in str(error)

    def test_with_reason(self):
        """Test error with reason."""
        error = ConditionCompilationError(
            condition="user.x ? document.y",
            rule_name="test-rule",
            reason="Unknown operator '?'"
        )
        assert "Unknown operator '?'" in str(error)

    def test_with_cause(self):
        """Test error with cause exception."""
        cause = ValueError("Invalid syntax")
        error = ConditionCompilationError(
            condition="bad condition",
            cause=cause
        )
        assert "ValueError" in str(error)
        assert "Invalid syntax" in str(error)

    def test_is_policy_error_subclass(self):
        """Test that ConditionCompilationError is a PolicyError subclass."""
        error = ConditionCompilationError(condition="x")
        assert isinstance(error, PolicyError)


class TestPolicyEngineConditionCompilation:
    """Test that PolicyEngine raises ConditionCompilationError for invalid conditions."""

    def test_valid_condition_no_error(self):
        """Test that valid conditions don't raise errors."""
        # Should not raise
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "valid-rule",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            }],
            "default": "deny"
        })
        assert len(policy.rules) == 1

    def test_invalid_operator_caught_at_parse_time(self):
        """Test that invalid operators are caught at Policy parse time by Pydantic."""
        from pydantic import ValidationError

        # Invalid operators are caught by Pydantic validators in Policy.from_dict
        with pytest.raises(ValidationError):
            Policy.from_dict({
                "version": "1",
                "rules": [{
                    "name": "bad-operator",
                    "allow": {
                        "conditions": ["user.x === document.y"]  # === not supported
                    }
                }],
                "default": "deny"
            })

    def test_exception_attributes(self):
        """Test ConditionCompilationError attributes are accessible."""
        cause = ValueError("test error")
        error = ConditionCompilationError(
            condition="user.x == y",
            rule_name="test-rule",
            reason="missing prefix",
            cause=cause
        )

        assert error.condition == "user.x == y"
        assert error.rule_name == "test-rule"
        assert error.reason == "missing prefix"
        assert error.cause is cause


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
