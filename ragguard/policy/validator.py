"""
Policy validation at load time.

This module validates policies when they are loaded, catching errors early
rather than at runtime. Validates:
- Semantic correctness
- Field references
- Unreachable rules
- Security issues (overly permissive policies)
- Performance concerns (overly complex policies)
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from ..exceptions import PolicyValidationError
from .models import Policy, Rule


class ValidationLevel(Enum):
    """Severity level for validation issues."""
    ERROR = "error"      # Blocks policy loading
    WARNING = "warning"  # Allows loading but logs warning
    INFO = "info"        # Informational message


@dataclass
class ValidationIssue:
    """Represents a validation issue found in a policy."""
    level: ValidationLevel
    rule_name: Optional[str]
    message: str
    suggestion: Optional[str] = None
    condition: Optional[str] = None

    def __str__(self) -> str:
        """Format issue for display."""
        parts = [f"[{self.level.value.upper()}]"]

        if self.rule_name:
            parts.append(f"Rule '{self.rule_name}':")

        parts.append(self.message)

        if self.condition:
            parts.append(f"\n  Condition: {self.condition}")

        if self.suggestion:
            parts.append(f"\n  Suggestion: {self.suggestion}")

        return " ".join(parts)


class PolicyValidator:
    """
    Validates policies at load time to catch errors early.

    Example:
        policy = Policy.from_dict(policy_dict)
        validator = PolicyValidator()
        issues = validator.validate(policy)

        if validator.has_errors(issues):
            for issue in issues:
                print(issue)
            raise PolicyValidationError("Policy has validation errors")
    """

    def __init__(
        self,
        strict: bool = False,
        max_rules: int = 100,
        max_conditions_per_rule: int = 20,
        warn_on_everyone: bool = True
    ):
        """
        Initialize policy validator.

        Args:
            strict: If True, treat warnings as errors
            max_rules: Maximum number of rules before warning
            max_conditions_per_rule: Maximum conditions per rule before warning
            warn_on_everyone: Warn about rules that allow everyone
        """
        self.strict = strict
        self.max_rules = max_rules
        self.max_conditions_per_rule = max_conditions_per_rule
        self.warn_on_everyone = warn_on_everyone

    def validate(self, policy: Policy) -> List[ValidationIssue]:
        """
        Validate a policy and return list of issues.

        Args:
            policy: Policy to validate

        Returns:
            List of validation issues (errors, warnings, info)
        """
        issues: List[ValidationIssue] = []

        # Basic structure validation
        issues.extend(self._validate_structure(policy))

        # Validate each rule
        for rule in policy.rules:
            issues.extend(self._validate_rule(rule, policy))

        # Check for rule interactions
        issues.extend(self._validate_rule_interactions(policy))

        # Performance checks
        issues.extend(self._validate_performance(policy))

        # Security checks
        issues.extend(self._validate_security(policy))

        return issues

    def has_errors(self, issues: List[ValidationIssue]) -> bool:
        """Check if issues list contains any errors."""
        if self.strict:
            return any(issue.level in (ValidationLevel.ERROR, ValidationLevel.WARNING)
                      for issue in issues)
        return any(issue.level == ValidationLevel.ERROR for issue in issues)

    def _validate_structure(self, policy: Policy) -> List[ValidationIssue]:
        """Validate basic policy structure."""
        issues = []

        # Check if policy has any rules
        if not policy.rules or len(policy.rules) == 0:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                rule_name=None,
                message="Policy has no rules",
                suggestion="Add at least one rule to control access"
            ))

        # Check for duplicate rule names
        rule_names = [rule.name for rule in policy.rules]
        duplicates = set([name for name in rule_names if rule_names.count(name) > 1])

        if duplicates:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                rule_name=None,
                message=f"Duplicate rule names found: {', '.join(duplicates)}",
                suggestion="Use unique names for each rule to avoid confusion"
            ))

        # Check default action
        if policy.default not in ("allow", "deny"):
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                rule_name=None,
                message=f"Invalid default action: '{policy.default}'",
                suggestion="Use 'allow' or 'deny' as default action"
            ))

        return issues

    def _validate_rule(self, rule: Rule, policy: Policy) -> List[ValidationIssue]:
        """Validate a single rule."""
        issues = []

        # Check if rule has any allow conditions
        allow = rule.allow
        has_user_check = allow.everyone or (allow.roles and len(allow.roles) > 0)
        has_conditions = allow.conditions and len(allow.conditions) > 0

        if not has_user_check and not has_conditions:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                rule_name=rule.name,
                message="Rule has no allow conditions (no everyone, roles, or conditions)",
                suggestion="Add at least one allow condition: everyone, roles, or conditions"
            ))

        # Validate conditions
        if has_conditions:
            issues.extend(self._validate_conditions(rule, allow.conditions))

        # Check match filters
        if rule.match:
            issues.extend(self._validate_match_filters(rule))

        return issues

    def _validate_conditions(
        self,
        rule: Rule,
        conditions: List[str]
    ) -> List[ValidationIssue]:
        """Validate rule conditions."""
        issues = []

        # Check number of conditions
        if len(conditions) > self.max_conditions_per_rule:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                rule_name=rule.name,
                message=f"Rule has {len(conditions)} conditions (>{self.max_conditions_per_rule})",
                suggestion="Consider splitting into multiple rules for better performance"
            ))

        for condition in conditions:
            issues.extend(self._validate_condition_syntax(rule, condition))

        return issues

    def _validate_condition_syntax(
        self,
        rule: Rule,
        condition: str
    ) -> List[ValidationIssue]:
        """Validate condition syntax and semantics."""
        issues = []

        # Check for common mistakes (single = instead of ==)
        # Exclude valid operators: ==, !=, <=, >=
        if "=" in condition and "==" not in condition and "!=" not in condition and "<=" not in condition and ">=" not in condition:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                rule_name=rule.name,
                message="Single '=' found in condition (use '==' for comparison)",
                condition=condition,
                suggestion="Use '==' for equality comparison, not '='"
            ))

        # Check for missing prefixes
        if not any(prefix in condition for prefix in ["user.", "document.", "EXISTS", "NOT EXISTS"]):
            # Could be a literal comparison, but warn anyway
            if not any(op in condition for op in ["==", "!=", "<", ">", "<=", ">="]):
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    rule_name=rule.name,
                    message="Condition doesn't reference user or document fields",
                    condition=condition,
                    suggestion="Use 'user.field' or 'document.field' to reference context"
                ))

        # Check for potential typos
        if "user,." in condition or "document,." in condition:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                rule_name=rule.name,
                message="Malformed field access (comma instead of dot)",
                condition=condition,
                suggestion="Use 'user.field' not 'user,.field'"
            ))

        # Check for None comparisons (common mistake after our security fix)
        if "== None" in condition or "!= None" in condition:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                rule_name=rule.name,
                message="Comparing with None literal (fields are auto-checked for None)",
                condition=condition,
                suggestion="Use 'EXISTS(field)' or 'NOT EXISTS(field)' instead"
            ))

        # Check for dangerous patterns
        if "__" in condition:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                rule_name=rule.name,
                message="Double underscore in condition (potential security risk)",
                condition=condition,
                suggestion="Avoid __ in field names (reserved for Python internals)"
            ))

        return issues

    def _validate_match_filters(self, rule: Rule) -> List[ValidationIssue]:
        """Validate match filters on a rule."""
        issues: List[ValidationIssue] = []

        if not rule.match:
            return issues

        # Check for empty match
        if not rule.match:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                rule_name=rule.name,
                message="Rule has empty match filter",
                suggestion="Remove empty match filter or add match conditions"
            ))

        return issues

    def _validate_rule_interactions(self, policy: Policy) -> List[ValidationIssue]:
        """Check for problematic interactions between rules."""
        issues = []

        # Check if all rules are unreachable (only happens with deny default)
        if policy.default == "deny":
            all_unreachable = all(
                not rule.allow.everyone
                and (not rule.allow.roles or len(rule.allow.roles) == 0)
                and (not rule.allow.conditions or len(rule.allow.conditions) == 0)
                for rule in policy.rules
            )

            if all_unreachable:
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    rule_name=None,
                    message="All rules are unreachable (no documents will be accessible)",
                    suggestion="Add at least one rule with allow conditions"
                ))

        # Check for overly broad early rules that shadow later rules
        for i, rule in enumerate(policy.rules[:-1]):
            if rule.allow.everyone and not rule.match:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    rule_name=rule.name,
                    message=f"Rule allows everyone without match filter, shadows {len(policy.rules) - i - 1} later rules",
                    suggestion="Add match filter or move rule to end of policy"
                ))
                break  # Only warn once

        return issues

    def _validate_performance(self, policy: Policy) -> List[ValidationIssue]:
        """Check for performance concerns."""
        issues = []

        # Check total number of rules
        if len(policy.rules) > self.max_rules:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                rule_name=None,
                message=f"Policy has {len(policy.rules)} rules (>{self.max_rules})",
                suggestion="Consider consolidating rules for better performance"
            ))

        # Check for complex nested conditions
        for rule in policy.rules:
            if rule.allow.conditions:
                for condition in rule.allow.conditions:
                    # Count operators as proxy for complexity
                    operator_count = sum(
                        condition.count(op)
                        for op in ["AND", "OR", "==", "!=", "<", ">", "in", "not in"]
                    )

                    if operator_count > 5:
                        issues.append(ValidationIssue(
                            level=ValidationLevel.INFO,
                            rule_name=rule.name,
                            message=f"Complex condition with {operator_count} operators",
                            condition=condition,
                            suggestion="Consider splitting into multiple simpler conditions"
                        ))

        return issues

    def _validate_security(self, policy: Policy) -> List[ValidationIssue]:
        """Check for security concerns."""
        issues = []

        # Warn about overly permissive policies
        if self.warn_on_everyone:
            everyone_rules = [
                rule for rule in policy.rules
                if rule.allow.everyone and not rule.match
            ]

            if everyone_rules:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    rule_name=everyone_rules[0].name,
                    message=f"{len(everyone_rules)} rule(s) allow everyone without restrictions",
                    suggestion="Add match filters or conditions to restrict access"
                ))

        # Check for allow-by-default (less secure)
        if policy.default == "allow":
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                rule_name=None,
                message="Policy uses 'allow' as default (less secure)",
                suggestion="Use 'deny' as default and explicitly allow access"
            ))

        # Check for rules with no user identity check
        for rule in policy.rules:
            if rule.allow.conditions and not rule.allow.everyone and not rule.allow.roles:
                # Check if any condition references user context
                has_user_ref = any(
                    "user." in condition
                    for condition in rule.allow.conditions
                )

                if not has_user_ref:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        rule_name=rule.name,
                        message="Rule conditions don't reference user context",
                        suggestion="Add user identity checks for better access control"
                    ))

        return issues


def validate_policy(
    policy: Policy,
    strict: bool = False,
    raise_on_error: bool = True
) -> List[ValidationIssue]:
    """
    Convenience function to validate a policy.

    Args:
        policy: Policy to validate
        strict: Treat warnings as errors
        raise_on_error: Raise exception if errors found

    Returns:
        List of validation issues

    Raises:
        PolicyValidationError: If errors found and raise_on_error=True
    """
    validator = PolicyValidator(strict=strict)
    issues = validator.validate(policy)

    if raise_on_error and validator.has_errors(issues):
        error_messages = [str(issue) for issue in issues if issue.level == ValidationLevel.ERROR]
        raise PolicyValidationError(
            f"Policy validation failed with {len(error_messages)} error(s):\n" +
            "\n".join(error_messages)
        )

    return issues


def print_validation_issues(issues: List[ValidationIssue]) -> None:
    """Print validation issues in a readable format."""
    if not issues:
        print("✓ Policy validation passed with no issues")
        return

    errors = [i for i in issues if i.level == ValidationLevel.ERROR]
    warnings = [i for i in issues if i.level == ValidationLevel.WARNING]
    infos = [i for i in issues if i.level == ValidationLevel.INFO]

    if errors:
        print(f"\n❌ {len(errors)} Error(s):")
        for issue in errors:
            print(f"  {issue}")

    if warnings:
        print(f"\n⚠️  {len(warnings)} Warning(s):")
        for issue in warnings:
            print(f"  {issue}")

    if infos:
        print(f"\n💡 {len(infos)} Info:")
        for issue in infos:
            print(f"  {issue}")
