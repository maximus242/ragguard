"""
Policy evaluation engine.

Evaluates access control policies to determine if users can access documents
and generates database-specific filters for permission-aware search.
"""

import logging
import threading
from typing import Any, Callable, Dict, Optional, Tuple

from ..constants import (
    MAX_BACKEND_NAME_LENGTH,
)
from ..exceptions import ConditionCompilationError, PolicyEvaluationError
from ..filters.backends.arangodb import to_arangodb_filter

# Import graph filter builders
from ..filters.backends.neo4j import to_neo4j_filter
from ..filters.backends.neptune import to_neptune_filter
from ..filters.backends.tigergraph import to_tigergraph_filter

# Import all filter builders at module level to avoid repeated imports
from ..filters.builder import (
    to_azure_search_filter,
    to_chromadb_filter,
    to_elasticsearch_filter,
    to_milvus_filter,
    to_pgvector_filter,
    to_pinecone_filter,
    to_qdrant_filter,
    to_weaviate_filter,
)
from ..filters.cache import (
    CacheKeyBuilder,
    FilterCache,
    compute_policy_hash,
    extract_user_fields_from_policy,
)
from ..types import FilterResult
from .compiler import CompiledConditionEvaluator, ConditionCompiler
from .models import AllowConditions, Policy, Rule

# Module logger
_logger = logging.getLogger(__name__)


def _is_deny_all_filter(filter_obj: Any, backend: str) -> bool:
    """
    Check if a filter represents a deny-all pattern.

    SECURITY: This function detects filter patterns that match nothing,
    which are used when access should be completely denied.

    Args:
        filter_obj: The filter object to check
        backend: The backend name

    Returns:
        True if this is a deny-all filter pattern
    """
    # Check for tuple pattern (pgvector)
    if isinstance(filter_obj, tuple) and len(filter_obj) >= 1:
        sql_clause = filter_obj[0] if isinstance(filter_obj[0], str) else ""
        if "FALSE" in sql_clause.upper() or "1=0" in sql_clause or "1 = 0" in sql_clause:
            return True

    # Check for string pattern (milvus, azure_search)
    if isinstance(filter_obj, str):
        upper = filter_obj.upper()
        if "FALSE" in upper or "1=0" in filter_obj or "1==0" in filter_obj or "1 == 0" in filter_obj:
            return True

    # Check for dict pattern (chromadb, pinecone, elasticsearch)
    if isinstance(filter_obj, dict):
        # Check for deny-all sentinel fields (both patterns used in codebase)
        deny_fields = ("__deny_all__", "__ragguard_deny_all__")
        for field in deny_fields:
            if field in filter_obj:
                return True
        if "$and" in filter_obj:
            # Check nested $and for deny pattern
            and_list = filter_obj.get("$and", [])
            if isinstance(and_list, list):
                for item in and_list:
                    if isinstance(item, dict):
                        for field in deny_fields:
                            if field in item:
                                return True
        # Check elasticsearch/opensearch bool pattern
        if "bool" in filter_obj:
            bool_clause = filter_obj.get("bool", {})
            if isinstance(bool_clause, dict):
                must_clause = bool_clause.get("must", [])
                if isinstance(must_clause, list):
                    for item in must_clause:
                        if isinstance(item, dict) and "term" in item:
                            term = item.get("term", {})
                            for field in deny_fields:
                                if field in term:
                                    return True

    return False


def _validate_filter_result(
    filter_obj: Any,
    backend: str,
    user: Dict[str, Any],
    policy_default: str
) -> Tuple[Any, Optional[str]]:
    """
    Validate filter builder result and detect potential issues.

    SECURITY: This validation helps detect ambiguous filter results that
    could lead to unintended access. It logs warnings for suspicious patterns.

    Args:
        filter_obj: The filter returned by the filter builder
        backend: The backend name
        user: User context (for logging)
        policy_default: The policy default ("allow" or "deny")

    Returns:
        Tuple of (validated_filter, warning_message)
        warning_message is None if no issues detected
    """
    warning = None

    # Check for None filter with deny default (potential security issue)
    if filter_obj is None:
        if policy_default == "deny":
            # None usually means "allow all" but policy default is deny
            # This could be intentional (e.g., admin user) or a bug
            _logger.debug(
                "Filter builder returned None (allow all) with deny default policy. "
                "User: %s, Backend: %s. This may be intentional for privileged users.",
                user.get("id", "unknown"),
                backend
            )

    # Check for empty dict (ambiguous in some backends)
    elif isinstance(filter_obj, dict) and len(filter_obj) == 0:
        warning = (
            f"Filter builder returned empty dict for backend '{backend}'. "
            f"Empty dict semantics vary by backend. User: {user.get('id', 'unknown')}"
        )
        _logger.warning(warning)

    # Check for empty list (ambiguous)
    elif isinstance(filter_obj, list) and len(filter_obj) == 0:
        warning = (
            f"Filter builder returned empty list for backend '{backend}'. "
            f"Empty list semantics are ambiguous. User: {user.get('id', 'unknown')}"
        )
        _logger.warning(warning)

    return filter_obj, warning


# Filter builder registry for O(1) lookup
_FILTER_BUILDERS: Dict[str, Callable] = {
    # Vector databases
    "qdrant": to_qdrant_filter,
    "pgvector": to_pgvector_filter,
    "weaviate": to_weaviate_filter,
    "pinecone": to_pinecone_filter,
    "chromadb": to_chromadb_filter,
    "milvus": to_milvus_filter,
    "elasticsearch": to_elasticsearch_filter,
    "opensearch": to_elasticsearch_filter,  # OpenSearch uses same format
    "azure_search": to_azure_search_filter,
    "azure_cognitive_search": to_azure_search_filter,  # Legacy name
    "faiss": None,  # FAISS doesn't support native metadata filtering
    # Graph databases
    "neo4j": to_neo4j_filter,
    "neptune": to_neptune_filter,
    "tigergraph": to_tigergraph_filter,
    "arangodb": to_arangodb_filter,
}


class PolicyEngine:
    """Evaluates access control policies and generates database filters."""

    def __init__(
        self,
        policy: Policy,
        backend: Optional[str] = None,
        enable_filter_cache: bool = True,
        filter_cache_size: int = 1000
    ):
        """
        Initialize the policy engine.

        Args:
            policy: The access control policy to enforce
            backend: Optional backend to optimize for. If None, supports all backends.
            enable_filter_cache: Whether to enable filter caching (default: True)
            filter_cache_size: Maximum number of cached filters (default: 1000)
        """
        self.policy = policy
        self._backend = backend

        # Initialize filter cache
        self._filter_cache = FilterCache(max_size=filter_cache_size) if enable_filter_cache else None

        # Pre-compute policy hash for cache keys
        self._policy_hash = compute_policy_hash(policy)

        # Thread-local storage for tracking last cache operation result
        # This avoids race conditions when detecting cache hits
        self._thread_local = threading.local()

        # Compile all conditions at initialization time for performance
        # This avoids string parsing on every condition evaluation
        # Supports both simple conditions and complex OR/AND expressions
        self._compiled_conditions = {}  # rule_index -> list of (CompiledCondition | CompiledExpression)
        for i, rule in enumerate(policy.rules):
            if rule.allow.conditions:
                compiled = []
                for condition_str in rule.allow.conditions:
                    try:
                        # Use compile_expression to support OR/AND logic
                        compiled_node = ConditionCompiler.compile_expression(condition_str)
                        compiled.append(compiled_node)
                    except (ValueError, TypeError, RecursionError) as e:
                        # Catch specific parsing/compilation errors
                        raise ConditionCompilationError(
                            condition=condition_str,
                            rule_name=rule.name,
                            reason=str(e),
                            cause=e
                        )
                self._compiled_conditions[i] = compiled

        # Pre-convert role lists to sets for O(1) lookup instead of O(n)
        # This avoids repeated list iteration during role checks
        self._role_sets = {}  # rule_index -> set of roles
        for i, rule in enumerate(policy.rules):
            if rule.allow.roles:
                self._role_sets[i] = set(rule.allow.roles)

        # Extract user fields referenced in policy for efficient cache keys
        # SECURITY: Pass compiled conditions for more accurate field extraction
        # This prevents cache poisoning by ensuring all relevant fields are captured
        self._relevant_user_fields = extract_user_fields_from_policy(
            policy,
            compiled_conditions=self._compiled_conditions
        )

    def evaluate(self, user: dict[str, Any], document: dict[str, Any]) -> bool:
        """
        Evaluate if a user can access a specific document.

        Args:
            user: User context (id, roles, department, etc.)
            document: Document metadata

        Returns:
            True if access is allowed, False otherwise

        Policy semantics:
        - Explicit rules (with match conditions) are checked first
        - If an explicit rule matches the document:
          - If user is allowed: grant access
          - If user is not allowed: deny access (explicit deny)
        - If no explicit rules match, check catch-all rules (without match conditions)
        - If no rules grant access, apply the default policy
        """
        # Check all rules with their indices for compiled condition lookup
        for i, rule in enumerate(self.policy.rules):
            # Check explicit rules first (rules with match conditions)
            if rule.match is not None:
                if not self._document_matches_rule(document, rule):
                    continue

                # This explicit rule matches the document
                if self._user_allowed(user, document, rule.allow, i):
                    return True  # Explicit allow
                else:
                    return False  # Explicit deny - don't check other rules

        # No explicit rules matched, check catch-all rules (without match conditions)
        for i, rule in enumerate(self.policy.rules):
            if rule.match is None:
                if self._user_allowed(user, document, rule.allow, i):
                    return True  # Catch-all allow

        # No rules granted access, apply default
        return self.policy.default == "allow"

    def evaluate_with_explanation(
        self,
        user: dict[str, Any],
        document: dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate access and return detailed explanation of the decision.

        This is useful for debugging complex policies and understanding why
        access was granted or denied.

        Args:
            user: User context
            document: Document metadata

        Returns:
            Dictionary with:
            - decision: "allow" or "deny"
            - reason: Human-readable explanation
            - rules_evaluated: List of rule evaluations with details
            - matched_rule: Name of the rule that granted/denied access (if any)
            - default_applied: Whether the default policy was applied

        Example:
            >>> result = engine.evaluate_with_explanation(user, document)
            >>> print(result["decision"])  # "allow" or "deny"
            >>> print(result["reason"])    # "Rule 'admin-access' granted access"
            >>> for rule_eval in result["rules_evaluated"]:
            ...     print(f"Rule: {rule_eval['name']}, Matched: {rule_eval['matched']}")
        """
        explanation = {
            "decision": None,
            "reason": None,
            "rules_evaluated": [],
            "matched_rule": None,
            "default_applied": False
        }

        # Check all rules with their indices
        for i, rule in enumerate(self.policy.rules):
            rule_eval: Dict[str, Any] = {
                "name": rule.name,
                "type": "explicit" if rule.match is not None else "catch-all",
                "match_conditions": rule.match,
                "matched_document": False,
                "user_allowed": None,
                "condition_details": []
            }

            # Check explicit rules first
            if rule.match is not None:
                match_result = self._explain_document_match(document, rule)
                rule_eval["matched_document"] = match_result["matched"]
                rule_eval["match_details"] = match_result["details"]

                if not match_result["matched"]:
                    rule_eval["skipped"] = True
                    rule_eval["skip_reason"] = "Document does not match rule's match conditions"
                    explanation["rules_evaluated"].append(rule_eval)
                    continue

                # Document matches, check user access
                user_result = self._explain_user_allowed(user, document, rule.allow, i)
                rule_eval["user_allowed"] = user_result["allowed"]
                rule_eval["allow_details"] = user_result["details"]
                explanation["rules_evaluated"].append(rule_eval)

                if user_result["allowed"]:
                    explanation["decision"] = "allow"
                    explanation["reason"] = f"Rule '{rule.name}' granted access"
                    explanation["matched_rule"] = rule.name
                    return explanation
                else:
                    explanation["decision"] = "deny"
                    explanation["reason"] = f"Rule '{rule.name}' explicitly denied access"
                    explanation["matched_rule"] = rule.name
                    return explanation

        # No explicit rules matched, check catch-all rules
        for i, rule in enumerate(self.policy.rules):
            if rule.match is None:
                rule_eval = {
                    "name": rule.name,
                    "type": "catch-all",
                    "matched_document": True,
                    "condition_details": []
                }

                user_result = self._explain_user_allowed(user, document, rule.allow, i)
                rule_eval["user_allowed"] = user_result["allowed"]
                rule_eval["allow_details"] = user_result["details"]
                explanation["rules_evaluated"].append(rule_eval)

                if user_result["allowed"]:
                    explanation["decision"] = "allow"
                    explanation["reason"] = f"Catch-all rule '{rule.name}' granted access"
                    explanation["matched_rule"] = rule.name
                    return explanation

        # No rules granted access, apply default
        explanation["default_applied"] = True
        explanation["decision"] = self.policy.default
        explanation["reason"] = f"No rules granted access, default policy '{self.policy.default}' applied"

        return explanation

    def _explain_document_match(
        self,
        document: dict[str, Any],
        rule: Rule
    ) -> Dict[str, Any]:
        """Explain why a document does or doesn't match a rule."""
        if rule.match is None:
            return {"matched": True, "details": "No match conditions (catches all documents)"}

        details = []
        matched = True

        for key, expected_value in rule.match.items():
            doc_value = self._get_nested_value(document, key)

            if isinstance(expected_value, list):
                if doc_value not in expected_value:
                    matched = False
                    details.append({
                        "field": key,
                        "expected": f"in {expected_value}",
                        "actual": doc_value,
                        "result": False
                    })
                else:
                    details.append({
                        "field": key,
                        "expected": f"in {expected_value}",
                        "actual": doc_value,
                        "result": True
                    })
            else:
                if doc_value != expected_value:
                    matched = False
                    details.append({
                        "field": key,
                        "expected": expected_value,
                        "actual": doc_value,
                        "result": False
                    })
                else:
                    details.append({
                        "field": key,
                        "expected": expected_value,
                        "actual": doc_value,
                        "result": True
                    })

        return {"matched": matched, "details": details}

    def _explain_user_allowed(
        self,
        user: dict[str, Any],
        document: dict[str, Any],
        allow: AllowConditions,
        rule_index: int
    ) -> Dict[str, Any]:
        """Explain why a user is or isn't allowed access."""
        details: Dict[str, Any] = {
            "everyone": None,
            "roles": None,
            "conditions": []
        }

        # Check everyone flag
        if allow.everyone is True:
            details["everyone"] = {"specified": True, "result": True}
            user_allowed = True
        elif allow.everyone is False:
            details["everyone"] = {"specified": True, "result": False}
            user_allowed = False
        else:
            user_allowed = False

        # Check roles
        if allow.roles is not None and len(allow.roles) > 0:
            user_roles = user.get("roles", [])
            if user_roles is None:
                user_roles = []
            elif isinstance(user_roles, str):
                user_roles = [user_roles]

            if rule_index in self._role_sets:
                role_set = self._role_sets[rule_index]
                matched_roles = [role for role in user_roles if role in role_set]
                role_allowed = len(matched_roles) > 0
            else:
                matched_roles = [role for role in user_roles if role in allow.roles]
                role_allowed = len(matched_roles) > 0

            details["roles"] = {
                "required": list(allow.roles),
                "user_has": user_roles,
                "matched": matched_roles,
                "result": role_allowed
            }

            user_allowed = role_allowed

        # Check conditions
        condition_check_specified = allow.conditions is not None and len(allow.conditions) > 0
        condition_check_passed = False

        if condition_check_specified:
            if rule_index in self._compiled_conditions:
                compiled_nodes = self._compiled_conditions[rule_index]
                condition_check_passed = True

                for i, compiled_node in enumerate(compiled_nodes):
                    condition_str = allow.conditions[i] if i < len(allow.conditions) else str(compiled_node)
                    result = CompiledConditionEvaluator.evaluate_node(compiled_node, user, document)

                    details["conditions"].append({
                        "condition": condition_str,
                        "result": result,
                        "type": "compiled"
                    })

                    if not result:
                        condition_check_passed = False
            else:
                # Fallback to string parsing
                condition_check_passed = True
                for condition in allow.conditions:
                    result = self._evaluate_condition(condition, user, document)
                    details["conditions"].append({
                        "condition": condition,
                        "result": result,
                        "type": "string_parsed"
                    })

                    if not result:
                        condition_check_passed = False

        # Determine if user-level checks are specified
        user_check_specified = allow.everyone is True or (allow.roles is not None and len(allow.roles) > 0)

        # Apply AND logic
        if user_check_specified and condition_check_specified:
            allowed = user_allowed and condition_check_passed
            details["logic"] = "user_allowed AND conditions_passed"
        elif user_check_specified:
            allowed = user_allowed
            details["logic"] = "user_allowed only"
        elif condition_check_specified:
            allowed = condition_check_passed
            details["logic"] = "conditions only"
        else:
            allowed = False
            details["logic"] = "no allow conditions specified"

        return {"allowed": allowed, "details": details}

    def _document_matches_rule(self, document: dict[str, Any], rule: Rule) -> bool:
        """
        Check if a document matches a rule's match conditions.

        If no match conditions are specified, the rule matches all documents.
        """
        if rule.match is None:
            return True

        for key, expected_value in rule.match.items():
            doc_value = self._get_nested_value(document, key)

            # Handle list values (for "in" checks)
            if isinstance(expected_value, list):
                if doc_value not in expected_value:
                    return False
            else:
                if doc_value != expected_value:
                    return False

        return True

    def _user_allowed(
        self,
        user: dict[str, Any],
        document: dict[str, Any],
        allow: AllowConditions,
        rule_index: int
    ) -> bool:
        """
        Check if a user satisfies allow conditions.

        When multiple allow types are specified (roles, conditions), they are
        combined with AND logic - all must be satisfied.

        Args:
            user: User context
            document: Document metadata
            allow: Allow conditions from the rule
            rule_index: Index of the rule (for compiled condition lookup)
        """

        # Track which checks are specified and which pass
        condition_check_specified = allow.conditions is not None and len(allow.conditions) > 0
        condition_check_passed = False

        # Check user access (either everyone flag or roles)
        user_allowed = False

        # Check "everyone" flag - means all users are allowed
        if allow.everyone is True:
            user_allowed = True
        # Check role-based access using pre-converted role sets
        elif allow.roles is not None and len(allow.roles) > 0:
            user_roles = user.get("roles", [])
            # Handle None roles (treat as empty list)
            if user_roles is None:
                user_roles = []
            elif isinstance(user_roles, str):
                user_roles = [user_roles]

            # Use pre-converted role set for efficient set intersection
            if rule_index in self._role_sets:
                role_set = self._role_sets[rule_index]
                # Convert user_roles to set and use intersection for O(min(n,m)) performance
                # This is faster than any() which is O(n*m) in worst case
                user_allowed = bool(set(user_roles) & role_set)
            else:
                # Fallback to list check (shouldn't happen if pre-conversion succeeded)
                user_allowed = any(role in allow.roles for role in user_roles)

        # Check custom conditions using compiled conditions/expressions
        if condition_check_specified:
            # Use compiled conditions if available (performance optimization)
            if rule_index in self._compiled_conditions:
                compiled_nodes = self._compiled_conditions[rule_index]
                condition_check_passed = True
                for compiled_node in compiled_nodes:
                    # Use evaluate_node to support both CompiledCondition and CompiledExpression
                    if not CompiledConditionEvaluator.evaluate_node(compiled_node, user, document):
                        condition_check_passed = False
                        break
            else:
                # Fallback to string parsing (shouldn't happen if compilation succeeded)
                condition_check_passed = True
                for condition in allow.conditions:
                    if not self._evaluate_condition(condition, user, document):
                        condition_check_passed = False
                        break

        # Determine if user-level checks are specified
        user_check_specified = allow.everyone is True or (allow.roles is not None and len(allow.roles) > 0)

        # Case 1: Both user checks and conditions specified - apply AND logic
        if user_check_specified and condition_check_specified:
            return user_allowed and condition_check_passed

        # Case 2: Only user checks specified (no conditions)
        if user_check_specified and not condition_check_specified:
            return user_allowed

        # Case 3: Only conditions specified (no user checks)
        if not user_check_specified and condition_check_specified:
            return condition_check_passed

        # Case 4: Neither specified - deny (no way to grant access)
        return False

    def _evaluate_condition(
        self,
        condition: str,
        user: dict[str, Any],
        document: dict[str, Any]
    ) -> bool:
        """
        Evaluate a single condition expression.

        Supports:
        - user.field == document.field
        - user.field != value
        - user.field in document.field (where document.field is a list)
        - value not in document.field
        """
        condition = condition.strip()

        # Parse the condition
        # Supported operators: ==, !=, in, not in
        if " not in " in condition:
            left, right = condition.split(" not in ", 1)
            left = left.strip()
            right = right.strip()
            left_value = self._resolve_value(left, user, document)
            right_value = self._resolve_value(right, user, document)

            if not isinstance(right_value, list):
                return False
            return left_value not in right_value

        elif " in " in condition:
            left, right = condition.split(" in ", 1)
            left = left.strip()
            right = right.strip()
            left_value = self._resolve_value(left, user, document)
            right_value = self._resolve_value(right, user, document)

            if not isinstance(right_value, list):
                return False
            return left_value in right_value

        elif "==" in condition:
            left, right = condition.split("==", 1)
            left = left.strip()
            right = right.strip()
            left_value = self._resolve_value(left, user, document)
            right_value = self._resolve_value(right, user, document)

            # Security: Don't allow None == None to grant access
            # Missing fields should deny access, not grant it
            if left_value is None or right_value is None:
                return False

            return left_value == right_value

        elif "!=" in condition:
            left, right = condition.split("!=", 1)
            left = left.strip()
            right = right.strip()
            left_value = self._resolve_value(left, user, document)
            right_value = self._resolve_value(right, user, document)

            # Security: Missing fields should deny access, not grant it
            # If either value is None, return False (deny access)
            # Use EXISTS() operator if you want to explicitly allow missing fields
            if left_value is None or right_value is None:
                return False

            return left_value != right_value

        else:
            raise PolicyEvaluationError(
                f"Unsupported condition operator in: {condition}"
            )

    def _resolve_value(
        self,
        expr: str,
        user: dict[str, Any],
        document: dict[str, Any]
    ) -> Any:
        """
        Resolve a value expression to its actual value.

        Supports:
        - user.field -> user context
        - document.field -> document metadata
        - Literal strings (quoted)
        - Literal numbers
        """
        expr = expr.strip()

        # Handle user context
        if expr.startswith("user."):
            field = expr[5:]  # Remove "user." prefix
            return self._get_nested_value(user, field)

        # Handle document context
        if expr.startswith("document."):
            field = expr[9:]  # Remove "document." prefix
            return self._get_nested_value(document, field)

        # Handle literal strings (quoted)
        if (expr.startswith('"') and expr.endswith('"')) or \
           (expr.startswith("'") and expr.endswith("'")):
            return expr[1:-1]

        # Handle literal numbers
        try:
            # Check for float indicators: decimal point or scientific notation (e/E)
            if "." in expr or "e" in expr.lower():
                return float(expr)
            return int(expr)
        except ValueError:
            pass

        # Handle boolean literals
        if expr.lower() == "true":
            return True
        if expr.lower() == "false":
            return False

        # Handle list literals: ['a', 'b', 'c']
        if expr.startswith('[') and expr.endswith(']'):
            content = expr[1:-1].strip()
            if not content:
                return []

            # Split by comma and parse each item
            items = []
            for item in content.split(','):
                item = item.strip()
                if item:
                    # Recursively resolve each item
                    items.append(self._resolve_value(item, user, document))
            return items

        # If nothing else matches, treat as literal string
        return expr

    def _get_nested_value(self, obj: dict[str, Any], key: str) -> Any:
        """
        Get a value from a nested dictionary using dot notation.

        Example: "metadata.team" -> obj["metadata"]["team"]
        """
        keys = key.split(".")
        value = obj

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return None
            else:
                return None

        return value

    def to_filter(self, user: dict[str, Any], backend: str) -> Any:
        """
        Generate a database-specific filter for permission-aware search.

        This is the key innovation: instead of filtering after retrieval,
        we push the permission logic into the vector search itself.

        With caching enabled, filters are cached by (backend, policy, user context)
        to avoid rebuilding on every query. This provides significant performance
        improvements for repeated queries.

        Args:
            user: User context
            backend: Database backend ("qdrant", "pgvector", "weaviate", "pinecone", "chromadb", "faiss")

        Returns:
            Database-specific filter object
        """
        # Check cache first (if enabled)
        if self._filter_cache is not None:
            cache_key = CacheKeyBuilder.build_key(
                backend=backend,
                policy_hash=self._policy_hash,
                user=user,
                relevant_user_fields=self._relevant_user_fields
            )

            cached_filter = self._filter_cache.get(cache_key)
            if cached_filter is not None:
                # Mark this as a cache hit in thread-local storage
                self._thread_local.last_was_cache_hit = True
                return cached_filter

        # Cache miss or caching disabled - build filter
        # Mark this as a cache miss in thread-local storage
        self._thread_local.last_was_cache_hit = False
        filter_obj = self._build_filter(backend, user)

        # Store in cache (if enabled)
        if self._filter_cache is not None:
            self._filter_cache.set(cache_key, filter_obj)

        return filter_obj

    def _build_filter(self, backend: str, user: dict[str, Any]) -> Any:
        """
        Build a database-specific filter.

        This is separated from to_filter() to allow caching logic
        to wrap the actual filter building.

        Uses the pre-loaded filter builder registry for O(1) lookup
        without dynamic imports.

        Args:
            backend: Database backend
            user: User context

        Returns:
            Database-specific filter object
        """
        # Validate backend name before lookup
        if not backend or not isinstance(backend, str):
            raise PolicyEvaluationError("Backend must be a non-empty string")

        # Validate backend name length and characters to prevent injection
        if len(backend) > MAX_BACKEND_NAME_LENGTH:
            raise PolicyEvaluationError(
                f"Backend name too long: {len(backend)} chars (max {MAX_BACKEND_NAME_LENGTH})"
            )

        # Allow only alphanumeric, underscore, and hyphen
        if not all(c.isalnum() or c in ('_', '-') for c in backend):
            raise PolicyEvaluationError(
                f"Invalid backend name: '{backend}'. Only alphanumeric, underscore, and hyphen allowed"
            )

        # Use registry for O(1) lookup (no dynamic imports)
        if backend not in _FILTER_BUILDERS:
            from ..errors import SUPPORTED_BACKENDS, unsupported_backend_error
            raise PolicyEvaluationError(
                unsupported_backend_error(backend, SUPPORTED_BACKENDS, "filter generation")
            )

        builder = _FILTER_BUILDERS[backend]

        # FAISS doesn't support native metadata filtering
        if builder is None:
            return None

        filter_obj = builder(self.policy, user)

        # Validate the filter result for potential issues
        _validate_filter_result(filter_obj, backend, user, self.policy.default)

        return filter_obj

    def to_filter_result(self, user: dict[str, Any], backend: str) -> FilterResult:
        """
        Generate a database-specific filter with explicit result semantics.

        This method is similar to to_filter() but returns a FilterResult
        that makes the filter's meaning explicit (allow all, deny all, or conditional).

        SECURITY: Use this method when you need to distinguish between
        "no filter needed" (allow all) and "match nothing" (deny all).

        Args:
            user: User context
            backend: Database backend

        Returns:
            FilterResult with explicit semantics:
            - ALLOW_ALL: No filter needed, user can access all documents
            - DENY_ALL: Match nothing, user should see no results
            - CONDITIONAL: Apply the filter to restrict access
        """
        filter_obj = self.to_filter(user, backend)

        # Determine the result type based on filter value
        if filter_obj is None:
            # None typically means "allow all" in most backends
            return FilterResult.allow_all()

        # Check if this is a deny-all pattern
        if _is_deny_all_filter(filter_obj, backend):
            return FilterResult.deny_all(reason="No matching rules grant access")

        # Normal conditional filter
        return FilterResult.conditional(filter_obj)

    def is_deny_all_filter(self, filter_obj: Any, backend: str) -> bool:
        """
        Check if a filter represents a deny-all pattern.

        This is useful for retrievers that need to detect when the policy
        would deny all access and skip the database query entirely.

        Args:
            filter_obj: The filter object to check
            backend: The backend name

        Returns:
            True if this filter would match no documents
        """
        return _is_deny_all_filter(filter_obj, backend)

    def invalidate_cache(self) -> None:
        """
        Clear the filter cache.

        This should be called when the policy is updated or when you want
        to force filter regeneration. Useful for testing or policy changes.
        """
        if self._filter_cache is not None:
            self._filter_cache.invalidate_all()

    def get_cache_stats(self) -> Optional[dict[str, Any]]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache hit rate, size, etc., or None if caching is disabled.
            Contains:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - hit_rate: Percentage of requests that hit the cache
            - size: Current number of entries in cache
            - max_size: Maximum cache size
        """
        if self._filter_cache is not None:
            return self._filter_cache.get_stats()
        return None

    def was_last_call_cache_hit(self) -> bool:
        """
        Check if the last to_filter() call for this thread was a cache hit.

        This method is thread-safe and avoids race conditions by using
        thread-local storage. It should be called immediately after to_filter()
        to determine if the filter was retrieved from cache.

        Returns:
            True if the last to_filter() call hit the cache, False otherwise.
            Returns False if caching is disabled or if to_filter() was never called.
        """
        return getattr(self._thread_local, 'last_was_cache_hit', False)
