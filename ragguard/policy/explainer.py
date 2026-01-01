"""
Query Explainer for RAGGuard

Helps users understand WHY a document was or wasn't returned in search results.
Essential for debugging permission policies.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..policy.compiler import (
    CompiledCondition,
    CompiledConditionEvaluator,
    ConditionCompiler,
)
from ..policy.engine import PolicyEngine
from ..policy.models import Policy, Rule


@dataclass
class ConditionEvaluation:
    """Result of evaluating a single condition."""
    condition: str
    passed: bool
    reason: str
    user_value: Any = None
    document_value: Any = None


@dataclass
class RuleEvaluation:
    """Result of evaluating a single rule."""
    rule_name: str
    matched: bool  # Did the rule's match conditions apply?
    passed: bool  # Did the rule's allow conditions pass?
    reason: str
    condition_evaluations: List[ConditionEvaluation] = field(default_factory=list)
    roles_matched: Optional[bool] = None
    everyone_matched: Optional[bool] = None


@dataclass
class QueryExplanation:
    """Complete explanation of why a document was allowed or denied."""
    document_id: Any
    final_decision: str  # "ALLOW" or "DENY"
    reason: str
    rule_evaluations: List[RuleEvaluation] = field(default_factory=list)
    default_policy: str = "deny"

    def __str__(self) -> str:
        """Format explanation as human-readable text."""
        lines = []
        lines.append(f"Document '{self.document_id}': {self.final_decision}")
        lines.append("")
        lines.append(f"Reason: {self.reason}")
        lines.append("")

        if self.rule_evaluations:
            lines.append("Rule Evaluation:")
            for rule_eval in self.rule_evaluations:
                lines.append(f"  • Rule '{rule_eval.rule_name}':")

                if not rule_eval.matched:
                    lines.append(f"    ✗ Not matched: {rule_eval.reason}")
                    continue

                if rule_eval.passed:
                    lines.append(f"    ✓ PASSED: {rule_eval.reason}")
                else:
                    lines.append(f"    ✗ FAILED: {rule_eval.reason}")

                # Show role checks
                if rule_eval.roles_matched is not None:
                    if rule_eval.roles_matched:
                        lines.append("      ✓ User has required role")
                    else:
                        lines.append("      ✗ User missing required role")

                # Show everyone check
                if rule_eval.everyone_matched is not None:
                    if rule_eval.everyone_matched:
                        lines.append("      ✓ Rule allows everyone")

                # Show condition evaluations
                for cond_eval in rule_eval.condition_evaluations:
                    if cond_eval.passed:
                        lines.append(f"      ✓ {cond_eval.condition}")
                    else:
                        lines.append(f"      ✗ {cond_eval.condition}")

                    if cond_eval.user_value is not None or cond_eval.document_value is not None:
                        lines.append(f"        User: {cond_eval.user_value}")
                        lines.append(f"        Document: {cond_eval.document_value}")

                    if cond_eval.reason:
                        lines.append(f"        {cond_eval.reason}")

        lines.append("")
        lines.append(f"Default Policy: {self.default_policy}")

        return "\n".join(lines)


class QueryExplainer:
    """
    Explains why documents are allowed or denied access.

    Example:
        >>> explainer = QueryExplainer(policy)
        >>> explanation = explainer.explain(
        ...     user={"id": "alice", "department": "engineering"},
        ...     document={"id": "doc123", "department": "finance"}
        ... )
        >>> print(explanation)
        Document 'doc123': DENY

        Reason: No rules granted access, default policy is deny

        Rule Evaluation:
          • Rule 'dept-access':
            ✗ FAILED: user.department != document.department
              ✗ user.department == document.department
                User: engineering
                Document: finance

        Default Policy: deny
    """

    def __init__(self, policy: Policy):
        """
        Initialize query explainer.

        Args:
            policy: The policy to explain
        """
        self.policy = policy
        self.engine = PolicyEngine(policy)

    def explain(
        self,
        user: Dict[str, Any],
        document: Dict[str, Any],
        document_id: Optional[Any] = None
    ) -> QueryExplanation:
        """
        Explain why a document is allowed or denied for a user.

        Args:
            user: User context
            document: Document metadata
            document_id: Optional document ID (defaults to document.get('id'))

        Returns:
            QueryExplanation with detailed reasoning
        """
        if document_id is None:
            document_id = document.get('id', 'unknown')

        rule_evaluations = []
        any_rule_passed = False

        # Evaluate each rule
        for rule in self.policy.rules:
            rule_eval = self._evaluate_rule(rule, user, document)
            rule_evaluations.append(rule_eval)

            if rule_eval.matched and rule_eval.passed:
                any_rule_passed = True

        # Determine final decision
        if any_rule_passed:
            final_decision = "ALLOW"
            reason = "At least one rule granted access"
        else:
            final_decision = "DENY"
            if not rule_evaluations:
                reason = "No rules defined, default policy is deny"
            else:
                reason = f"No rules granted access, default policy is {self.policy.default}"

        return QueryExplanation(
            document_id=document_id,
            final_decision=final_decision,
            reason=reason,
            rule_evaluations=rule_evaluations,
            default_policy=self.policy.default
        )

    def _evaluate_rule(
        self,
        rule: Rule,
        user: Dict[str, Any],
        document: Dict[str, Any]
    ) -> RuleEvaluation:
        """Evaluate a single rule and explain the result."""

        # Check if rule matches (via match conditions)
        if rule.match:
            matched = self._check_match(rule.match, document)
            if not matched:
                return RuleEvaluation(
                    rule_name=rule.name,
                    matched=False,
                    passed=False,
                    reason=f"Document doesn't match rule criteria: {rule.match}"
                )
        else:
            matched = True

        # Rule matched, now check allow conditions
        condition_evaluations = []
        roles_matched = None
        everyone_matched = None

        # Check "everyone" flag
        if rule.allow.everyone:
            everyone_matched = True
            return RuleEvaluation(
                rule_name=rule.name,
                matched=True,
                passed=True,
                reason="Rule allows everyone",
                everyone_matched=True
            )

        # Check roles
        if rule.allow.roles:
            user_roles = user.get('roles', [])
            if user_roles is None:
                user_roles = []
            elif isinstance(user_roles, str):
                user_roles = [user_roles]

            roles_matched = any(role in user_roles for role in rule.allow.roles)

            if roles_matched:
                return RuleEvaluation(
                    rule_name=rule.name,
                    matched=True,
                    passed=True,
                    reason=f"User has required role: {user_roles}",
                    roles_matched=True
                )
            else:
                return RuleEvaluation(
                    rule_name=rule.name,
                    matched=True,
                    passed=False,
                    reason=f"User roles {user_roles} don't include any of {rule.allow.roles}",
                    roles_matched=False
                )

        # Check conditions
        if rule.allow.conditions:
            all_passed = True

            for condition_str in rule.allow.conditions:
                cond_eval = self._evaluate_condition(condition_str, user, document)
                condition_evaluations.append(cond_eval)

                if not cond_eval.passed:
                    all_passed = False

            if all_passed:
                return RuleEvaluation(
                    rule_name=rule.name,
                    matched=True,
                    passed=True,
                    reason=f"All {len(condition_evaluations)} conditions passed",
                    condition_evaluations=condition_evaluations
                )
            else:
                failed_count = sum(1 for c in condition_evaluations if not c.passed)
                return RuleEvaluation(
                    rule_name=rule.name,
                    matched=True,
                    passed=False,
                    reason=f"{failed_count} of {len(condition_evaluations)} conditions failed",
                    condition_evaluations=condition_evaluations
                )

        # No conditions, roles, or everyone - rule doesn't grant access
        return RuleEvaluation(
            rule_name=rule.name,
            matched=True,
            passed=False,
            reason="Rule has no allow conditions"
        )

    def _check_match(self, match: Dict[str, Any], document: Dict[str, Any]) -> bool:
        """Check if document matches the rule's match conditions."""
        for key, value in match.items():
            if document.get(key) != value:
                return False
        return True

    def _evaluate_condition(
        self,
        condition_str: str,
        user: Dict[str, Any],
        document: Dict[str, Any]
    ) -> ConditionEvaluation:
        """Evaluate a single condition and explain the result."""
        try:
            # Compile and evaluate the condition using static method
            compiled = ConditionCompiler.compile_expression(condition_str)
            passed = CompiledConditionEvaluator.evaluate_node(compiled, user, document)

            # Extract values for explanation
            user_value, doc_value = self._extract_values(compiled, user, document)

            if passed:
                reason = "Condition satisfied"
            else:
                reason = "Condition not satisfied"

            return ConditionEvaluation(
                condition=condition_str,
                passed=passed,
                reason=reason,
                user_value=user_value,
                document_value=doc_value
            )

        except Exception as e:
            return ConditionEvaluation(
                condition=condition_str,
                passed=False,
                reason=f"Error evaluating condition: {e}"
            )

    def _get_nested_value(self, obj: Dict[str, Any], path: str) -> Any:
        """Get a nested value from an object using dot notation."""
        parts = path.split(".")
        current = obj
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    def _extract_values(
        self,
        compiled: Any,
        user: Dict[str, Any],
        document: Dict[str, Any]
    ) -> tuple[Any, Any]:
        """Extract user and document values from a compiled condition for display."""
        from ..policy.compiler import ValueType

        if isinstance(compiled, CompiledCondition):
            # Resolve values using the compiled evaluator
            left_val = CompiledConditionEvaluator._resolve_value(compiled.left, user, document)
            right_val = CompiledConditionEvaluator._resolve_value(compiled.right, user, document) if compiled.right else None

            # Determine which is user and which is document based on value types
            if compiled.left.value_type == ValueType.USER_FIELD:
                return left_val, right_val
            elif compiled.left.value_type == ValueType.DOCUMENT_FIELD:
                return right_val, left_val

            return left_val, right_val

        return None, None


__all__ = [
    "ConditionEvaluation",
    "QueryExplainer",
    "QueryExplanation",
    "RuleEvaluation",
]
