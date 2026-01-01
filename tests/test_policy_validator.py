"""
Tests for policy validation at load time.
"""

import pytest

from ragguard import Policy
from ragguard.exceptions import PolicyValidationError
from ragguard.policy.validator import (
    PolicyValidator,
    ValidationIssue,
    ValidationLevel,
    print_validation_issues,
    validate_policy,
)


def test_valid_policy_no_issues():
    """Test that a valid policy has no validation issues."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "department-access",
            "allow": {
                "conditions": ["user.department == document.department"]
            }
        }],
        "default": "deny"
    }, validate=False)  # Skip validation in from_dict to test validator directly

    validator = PolicyValidator()
    issues = validator.validate(policy)

    # Should have no errors
    errors = [i for i in issues if i.level == ValidationLevel.ERROR]
    assert len(errors) == 0


def test_policy_with_no_rules():
    """Test warning for policy with no rules."""
    # Use model_construct to bypass Pydantic validation
    from ragguard.policy.models import AllowConditions, Rule
    policy = Policy.model_construct(version="1", rules=[], default="deny")

    validator = PolicyValidator()
    issues = validator.validate(policy)

    # Should have warning about no rules
    warnings = [i for i in issues if i.level == ValidationLevel.WARNING]
    assert any("no rules" in str(issue).lower() for issue in warnings)


def test_duplicate_rule_names():
    """Test warning for duplicate rule names."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "access", "allow": {"everyone": True}},
            {"name": "access", "allow": {"roles": ["admin"]}}
        ],
        "default": "deny"
    }, validate=False)

    validator = PolicyValidator()
    issues = validator.validate(policy)

    warnings = [i for i in issues if i.level == ValidationLevel.WARNING]
    assert any("duplicate" in str(issue).lower() for issue in warnings)


def test_rule_with_no_allow_conditions():
    """Test error for rule with no allow conditions."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "empty-rule",
            "allow": {}
        }],
        "default": "deny"
    }, validate=False)

    validator = PolicyValidator()
    issues = validator.validate(policy)

    errors = [i for i in issues if i.level == ValidationLevel.ERROR]
    assert any("no allow conditions" in str(issue).lower() for issue in errors)


def test_single_equals_in_condition():
    """Test that Pydantic validation catches single '=' instead of '=='."""
    # This is now caught at Pydantic level, not semantic validator level
    # Test that it's properly rejected
    from ragguard.exceptions import PolicyValidationError

    with pytest.raises((PolicyValidationError, Exception)) as exc_info:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "bad-condition",
                "allow": {
                    "conditions": ["user.role = 'admin'"]  # Wrong! Should be ==
                }
            }],
            "default": "deny"
        }, validate=False)

    # Verify error message mentions the issue
    assert "=" in str(exc_info.value).lower()


def test_malformed_field_access():
    """Test error for malformed field access (comma instead of dot)."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "malformed",
            "allow": {
                "conditions": ["user,.role == 'admin'"]  # Wrong!
            }
        }],
        "default": "deny"
    }, validate=False)

    validator = PolicyValidator()
    issues = validator.validate(policy)

    errors = [i for i in issues if i.level == ValidationLevel.ERROR]
    assert any("malformed" in str(issue).lower() for issue in errors)


def test_none_comparison_warning():
    """Test warning for comparing with None literal."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "none-check",
            "allow": {
                "conditions": ["user.status != None"]  # Should use EXISTS
            }
        }],
        "default": "deny"
    }, validate=False)

    validator = PolicyValidator()
    issues = validator.validate(policy)

    warnings = [i for i in issues if i.level == ValidationLevel.WARNING]
    assert any("none literal" in str(issue).lower() for issue in warnings)


def test_double_underscore_security_risk():
    """Test error for double underscore in field names."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "unsafe",
            "allow": {
                "conditions": ["user.__class__ == 'Admin'"]  # Security risk!
            }
        }],
        "default": "deny"
    }, validate=False)

    validator = PolicyValidator()
    issues = validator.validate(policy)

    errors = [i for i in issues if i.level == ValidationLevel.ERROR]
    assert any("double underscore" in str(issue).lower() for issue in errors)


def test_too_many_rules_warning():
    """Test warning for too many rules."""
    # Pydantic has a limit of 100 rules, so test with 50 rules and validator limit of 40
    rules = [
        {"name": f"rule-{i}", "allow": {"everyone": True}}
        for i in range(50)
    ]

    policy = Policy.from_dict({
        "version": "1",
        "rules": rules,
        "default": "deny"
    }, validate=False)

    validator = PolicyValidator(max_rules=40)
    issues = validator.validate(policy)

    warnings = [i for i in issues if i.level == ValidationLevel.WARNING]
    assert any("50 rules" in str(issue) for issue in warnings)


def test_too_many_conditions_per_rule():
    """Test warning for too many conditions in one rule."""
    conditions = [
        f"user.field{i} == document.field{i}"
        for i in range(25)
    ]

    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "complex-rule",
            "allow": {"conditions": conditions}
        }],
        "default": "deny"
    }, validate=False)

    validator = PolicyValidator(max_conditions_per_rule=20)
    issues = validator.validate(policy)

    warnings = [i for i in issues if i.level == ValidationLevel.WARNING]
    assert any("25 conditions" in str(issue) for issue in warnings)


def test_everyone_rule_without_match_filter():
    """Test warning for everyone rule without match filter."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "allow-all",
            "allow": {"everyone": True}
        }],
        "default": "deny"
    }, validate=False)

    validator = PolicyValidator(warn_on_everyone=True)
    issues = validator.validate(policy)

    warnings = [i for i in issues if i.level == ValidationLevel.WARNING]
    assert any("allow everyone" in str(issue).lower() for issue in warnings)


def test_everyone_rule_shadows_later_rules():
    """Test warning when everyone rule shadows later rules."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "allow-all", "allow": {"everyone": True}},
            {"name": "specific-rule", "allow": {"roles": ["admin"]}}  # Shadowed!
        ],
        "default": "deny"
    }, validate=False)

    validator = PolicyValidator()
    issues = validator.validate(policy)

    warnings = [i for i in issues if i.level == ValidationLevel.WARNING]
    assert any("shadows" in str(issue).lower() for issue in warnings)


def test_allow_by_default_warning():
    """Test warning for allow-by-default policy."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "test",
            "allow": {"everyone": True}
        }],
        "default": "allow"  # Less secure
    }, validate=False)

    validator = PolicyValidator()
    issues = validator.validate(policy)

    warnings = [i for i in issues if i.level == ValidationLevel.WARNING]
    assert any("allow" in str(issue).lower() and "default" in str(issue).lower()
               for issue in warnings)


def test_rule_without_user_reference():
    """Test warning for rule without user context reference."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "no-user-check",
            "allow": {
                "conditions": ["document.status == 'public'"]  # No user reference
            }
        }],
        "default": "deny"
    }, validate=False)

    validator = PolicyValidator()
    issues = validator.validate(policy)

    warnings = [i for i in issues if i.level == ValidationLevel.WARNING]
    assert any("don't reference user" in str(issue).lower() for issue in warnings)


def test_validate_policy_raises_on_error():
    """Test that validate_policy raises exception on errors."""
    # Create policy with semantic error (no allow conditions)
    from ragguard.policy.models import AllowConditions, Rule

    rule = Rule.model_construct(
        name="bad",
        allow=AllowConditions.model_construct()  # Empty allow
    )
    policy = Policy.model_construct(
        version="1",
        rules=[rule],
        default="deny"
    )

    with pytest.raises(PolicyValidationError):
        validate_policy(policy, strict=False, raise_on_error=True)


def test_validate_policy_no_raise():
    """Test that validate_policy doesn't raise when raise_on_error=False."""
    # Create policy with semantic error (no allow conditions)
    from ragguard.policy.models import AllowConditions, Rule

    rule = Rule.model_construct(
        name="bad",
        allow=AllowConditions.model_construct()  # Empty allow
    )
    policy = Policy.model_construct(
        version="1",
        rules=[rule],
        default="deny"
    )

    issues = validate_policy(policy, strict=False, raise_on_error=False)
    errors = [i for i in issues if i.level == ValidationLevel.ERROR]
    assert len(errors) > 0


def test_strict_mode_treats_warnings_as_errors():
    """Test that strict mode treats warnings as errors."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "everyone",
            "allow": {"everyone": True}
        }],
        "default": "deny"
    }, validate=False)

    validator = PolicyValidator(strict=True)
    issues = validator.validate(policy)

    # Should have warnings
    warnings = [i for i in issues if i.level == ValidationLevel.WARNING]
    assert len(warnings) > 0

    # has_errors should return True in strict mode
    assert validator.has_errors(issues) == True


def test_validation_issue_str_formatting():
    """Test that ValidationIssue formats nicely."""
    issue = ValidationIssue(
        level=ValidationLevel.ERROR,
        rule_name="test-rule",
        message="Test error message",
        condition="user.role == 'admin'",
        suggestion="Use double equals"
    )

    issue_str = str(issue)
    assert "ERROR" in issue_str
    assert "test-rule" in issue_str
    assert "Test error message" in issue_str
    assert "user.role == 'admin'" in issue_str
    assert "Use double equals" in issue_str


def test_print_validation_issues_no_issues(capsys):
    """Test print function with no issues."""
    print_validation_issues([])

    captured = capsys.readouterr()
    assert "no issues" in captured.out.lower()


def test_print_validation_issues_with_errors(capsys):
    """Test print function with errors."""
    issues = [
        ValidationIssue(
            level=ValidationLevel.ERROR,
            rule_name="bad-rule",
            message="Test error"
        )
    ]

    print_validation_issues(issues)

    captured = capsys.readouterr()
    assert "Error" in captured.out
    assert "bad-rule" in captured.out


def test_complex_policy_validation():
    """Test validation of a complex realistic policy."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "public-docs",
                "match": {"visibility": "public"},
                "allow": {"everyone": True}
            },
            {
                "name": "department-docs",
                "allow": {
                    "conditions": [
                        "user.department exists",
                        "document.department exists",
                        "user.department == document.department"
                    ]
                }
            },
            {
                "name": "admin-access",
                "allow": {
                    "roles": ["admin", "superuser"]
                }
            }
        ],
        "default": "deny"
    }, validate=False)

    validator = PolicyValidator()
    issues = validator.validate(policy)

    # Should have minimal issues (maybe some info/warnings)
    errors = [i for i in issues if i.level == ValidationLevel.ERROR]
    assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
