"""
Pydantic models for RAGGuard policies.

Defines the schema for access control policies that determine
which users can access which documents.
"""

import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from ..types import PolicyDict


class PolicyLimits:
    """Complexity limits to prevent DoS attacks."""

    # Maximum number of rules in a policy
    MAX_RULES = 100

    # Maximum number of conditions per rule
    MAX_CONDITIONS_PER_RULE = 100

    # Maximum size of a list literal in a condition (e.g., in ['a', 'b', ...])
    MAX_LIST_SIZE = 1000

    # Maximum byte size of a list literal (prevents huge single elements)
    MAX_LIST_SIZE_BYTES = 100_000  # 100KB per list

    # Maximum total conditions across all rules
    MAX_TOTAL_CONDITIONS = 1000

    # Maximum policy size in bytes (prevents huge policy files)
    MAX_POLICY_SIZE_BYTES = 1_000_000  # 1MB

    # Maximum nesting depth in match conditions
    MAX_NESTING_DEPTH = 10

    # OR/AND Expression Limits (v0.3.0)
    # Maximum nesting depth for OR/AND expressions (prevents stack overflow)
    MAX_EXPRESSION_DEPTH = 10

    # Maximum number of OR/AND branches in a single expression (prevents exponential blowup)
    MAX_EXPRESSION_BRANCHES = 50

    # Maximum total number of conditions in an OR/AND expression tree
    MAX_EXPRESSION_CONDITIONS = 100


class AllowConditions(BaseModel):
    """Defines who is allowed access under a rule."""

    everyone: Optional[bool] = None
    roles: Optional[list[str]] = None
    conditions: Optional[list[str]] = None

    @field_validator("conditions")
    @classmethod
    def validate_conditions(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        """Validate that conditions use supported operators and don't exceed limits."""
        if v is None:
            return v

        # Import here to avoid circular dependency
        from ragguard.policy.compiler import ConditionCompiler

        # Check condition count limit
        if len(v) > PolicyLimits.MAX_CONDITIONS_PER_RULE:
            raise ValueError(
                f"Too many conditions in rule: {len(v)} > {PolicyLimits.MAX_CONDITIONS_PER_RULE}. "
                f"This could cause performance issues or DoS attacks."
            )

        # Validate each condition using the compiler (which has better error messages)
        for condition in v:
            try:
                # This will raise ValueError with detailed error messages if invalid
                ConditionCompiler.compile_condition(condition)
            except ValueError:
                # Re-raise the error from the compiler (it has better messages)
                raise

            # Check list literal size (count commas in brackets)
            if " in " in condition and "[" in condition:
                # Extract the list part - find bracket positions
                try:
                    start_idx = condition.index("[")
                    end_idx = condition.rindex("]")
                except ValueError:
                    # .index()/.rindex() raise ValueError if not found
                    # This shouldn't happen since we check for "[" above,
                    # but skip validation if it does
                    continue

                list_part = condition[start_idx:end_idx+1]

                # Check byte size first (prevents huge single elements)
                list_size_bytes = len(list_part.encode('utf-8'))
                if list_size_bytes > PolicyLimits.MAX_LIST_SIZE_BYTES:
                    raise ValueError(
                        f"List literal too large in condition: {list_size_bytes} bytes > {PolicyLimits.MAX_LIST_SIZE_BYTES} bytes. "
                        f"This could cause performance issues or DoS attacks."
                    )

                # Count elements: for a list ['a', 'b', 'c'], there are 2 commas
                # So num_elements = comma_count + 1, unless it's empty
                if list_part == "[]":
                    num_elements = 0
                else:
                    # Count commas, but be careful of commas inside quoted strings
                    # For simplicity, just count all commas as a heuristic
                    comma_count = list_part.count(",")
                    num_elements = comma_count + 1

                if num_elements > PolicyLimits.MAX_LIST_SIZE:
                    raise ValueError(
                        f"List literal too large in condition: {num_elements} elements > {PolicyLimits.MAX_LIST_SIZE}. "
                        f"This could cause performance issues or DoS attacks."
                    )

        return v

    @property
    def is_empty(self) -> bool:
        """Check if no allow conditions are specified."""
        return (
            self.everyone is None
            and (self.roles is None or len(self.roles) == 0)
            and (self.conditions is None or len(self.conditions) == 0)
        )


class Rule(BaseModel):
    """A single access control rule."""

    name: str = Field(..., description="Human-readable name for the rule")
    match: Optional[dict[str, Any]] = Field(
        default=None,
        description="Conditions that documents must match for this rule to apply"
    )
    allow: AllowConditions = Field(
        ...,
        description="Who is allowed access when this rule matches"
    )

    @field_validator("match")
    @classmethod
    def validate_match(cls, v: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Validate match conditions are simple key-value pairs and check nesting depth."""
        if v is None:
            return v

        def check_nesting_depth(obj: Any, depth: int = 0) -> int:
            """Recursively check nesting depth."""
            if depth > PolicyLimits.MAX_NESTING_DEPTH:
                raise ValueError(
                    f"Match condition nesting too deep: {depth} > {PolicyLimits.MAX_NESTING_DEPTH}. "
                    f"This could cause performance issues or DoS attacks."
                )

            if isinstance(obj, dict):
                return max([check_nesting_depth(v, depth + 1) for v in obj.values()], default=depth)
            elif isinstance(obj, list):
                return max([check_nesting_depth(item, depth + 1) for item in obj], default=depth)
            else:
                return depth

        # Check nesting depth
        check_nesting_depth(v)

        # Match conditions should be simple key-value pairs
        # We'll support nested keys like "metadata.team" but values should be simple
        for key, value in v.items():
            if isinstance(value, (dict, list)):
                # Allow lists for "in" checks
                if not isinstance(value, list):
                    raise ValueError(
                        f"Match condition values must be simple types or lists, got {type(value)} for key '{key}'"
                    )
        return v


class Policy(BaseModel):
    """Complete access control policy."""

    version: str = Field(..., description="Policy format version")
    rules: list[Rule] = Field(..., description="List of access control rules")
    default: Literal["deny", "allow"] = Field(
        default="deny",
        description="Default action when no rules match"
    )

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Ensure version is supported."""
        if v != "1":
            raise ValueError(f"Unsupported policy version: {v}. Only version '1' is supported.")
        return v

    @field_validator("rules")
    @classmethod
    def validate_rules_not_empty(cls, v: list[Rule]) -> list[Rule]:
        """Ensure at least one rule is defined and check rule count limit."""
        if not v:
            raise ValueError("Policy must have at least one rule")

        # Check rule count limit
        if len(v) > PolicyLimits.MAX_RULES:
            raise ValueError(
                f"Too many rules in policy: {len(v)} > {PolicyLimits.MAX_RULES}. "
                f"This could cause performance issues or DoS attacks."
            )

        # Check total conditions across all rules
        total_conditions = 0
        for rule in v:
            if rule.allow.conditions:
                total_conditions += len(rule.allow.conditions)

        if total_conditions > PolicyLimits.MAX_TOTAL_CONDITIONS:
            raise ValueError(
                f"Too many total conditions in policy: {total_conditions} > {PolicyLimits.MAX_TOTAL_CONDITIONS}. "
                f"This could cause performance issues or DoS attacks."
            )

        return v

    @classmethod
    def from_dict(cls, data: PolicyDict, validate: bool = True) -> "Policy":
        """
        Create a Policy from a dictionary.

        This is a convenience method that validates policy size before parsing.

        Args:
            data: Dictionary containing policy specification
            validate: If True, runs semantic validation and prints warnings (default: True)

        Returns:
            Policy instance

        Raises:
            ValueError: If policy is too large
            PolicyValidationError: If validate=True and policy has errors
        """
        # Check policy size limit
        policy_json = json.dumps(data)
        policy_size = len(policy_json.encode('utf-8'))

        if policy_size > PolicyLimits.MAX_POLICY_SIZE_BYTES:
            raise ValueError(
                f"Policy too large: {policy_size} bytes > {PolicyLimits.MAX_POLICY_SIZE_BYTES} bytes. "
                f"This could cause performance issues or DoS attacks."
            )

        # Use Pydantic's model_validate which is the v2 way
        policy = cls.model_validate(data)

        # Run semantic validation if requested
        if validate:
            from .validator import print_validation_issues, validate_policy
            issues = validate_policy(policy, strict=False, raise_on_error=True)

            # Print warnings (errors would have raised exception)
            if issues:
                print_validation_issues(issues)

        return policy
