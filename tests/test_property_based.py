"""
Property-based tests using hypothesis to verify security invariants.

These tests use randomized inputs to discover edge cases and ensure
that security properties hold across a wide range of inputs.
"""

from unittest.mock import MagicMock

import pytest

# Skip all tests if hypothesis is not installed
pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

# Common settings for all hypothesis tests
HYPOTHESIS_SETTINGS = settings(
    max_examples=100,
    deadline=None,  # Disable deadline to avoid flaky tests
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture
    ]
)


# Custom strategies for generating test data
@st.composite
def user_context(draw):
    """Generate random user contexts."""
    user_id = draw(st.text(min_size=1, max_size=100))

    # Generate roles as list, None, or missing
    roles_type = draw(st.sampled_from(['list', 'none', 'missing', 'string', 'empty']))

    user = {"id": user_id}

    if roles_type == 'list':
        roles = draw(st.lists(st.text(min_size=0, max_size=50), max_size=10))
        user["roles"] = roles
    elif roles_type == 'none':
        user["roles"] = None
    elif roles_type == 'string':
        user["roles"] = draw(st.text(min_size=0, max_size=50))
    elif roles_type == 'empty':
        user["roles"] = []
    # 'missing' - don't add roles field

    # Optionally add department
    if draw(st.booleans()):
        user["department"] = draw(st.text(min_size=0, max_size=50))

    # Optionally add team
    if draw(st.booleans()):
        user["team"] = draw(st.text(min_size=0, max_size=50))

    return user


@st.composite
def document_metadata(draw):
    """Generate random document metadata."""
    doc = {"id": draw(st.text(min_size=1, max_size=100))}

    # Optionally add various fields
    if draw(st.booleans()):
        doc["type"] = draw(st.sampled_from(['public', 'private', 'internal', 'confidential']))

    if draw(st.booleans()):
        doc["department"] = draw(st.text(min_size=0, max_size=50))

    if draw(st.booleans()):
        doc["visibility"] = draw(st.sampled_from(['public', 'private']))

    if draw(st.booleans()):
        doc["owner"] = draw(st.text(min_size=0, max_size=50))

    return doc


@st.composite
def malicious_user(draw):
    """Generate user contexts with potential injection attempts."""
    base_user = draw(user_context())

    # Add various injection attempts
    injection_attempts = [
        ("__roles__", ["admin", "superuser"]),
        ("_roles", ["admin"]),
        ("__proto__", {"roles": ["admin"]}),
        ("constructor", {"prototype": {"roles": ["admin"]}}),
        ("user", {"roles": ["admin"]}),
        ("__allow__", True),
        ("admin", True),
    ]

    if draw(st.booleans()):
        key, value = draw(st.sampled_from(injection_attempts))
        base_user[key] = value

    return base_user


class TestSecurityInvariants:
    """Property-based tests for security invariants."""

    @pytest.fixture
    def deny_all_policy(self):
        """Policy that denies all access (requires impossible role)."""
        from ragguard.policy.models import Policy
        return Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "impossible_role",
                    "allow": {"roles": ["__impossible_role_that_nobody_has__"]}
                }
            ],
            "default": "deny"
        })

    @pytest.fixture
    def admin_only_policy(self):
        """Policy that only allows admin role."""
        from ragguard.policy.models import Policy
        return Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "admin_only",
                    "allow": {"roles": ["admin"]}
                }
            ],
            "default": "deny"
        })

    @pytest.fixture
    def dept_match_policy(self):
        """Policy that requires department match."""
        from ragguard.policy.models import Policy
        return Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "dept_match",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.department == document.department"]
                    }
                }
            ],
            "default": "deny"
        })

    @given(user=user_context(), doc=document_metadata())
    @HYPOTHESIS_SETTINGS
    def test_deny_all_policy_always_denies(self, deny_all_policy, user, doc):
        """INVARIANT: Empty policy with default deny should always deny."""
        from ragguard.policy.engine import PolicyEngine

        engine = PolicyEngine(deny_all_policy)
        result = engine.evaluate(user, doc)

        assert result is False, f"Deny-all policy granted access to user: {user}"

    @given(user=malicious_user(), doc=document_metadata())
    @HYPOTHESIS_SETTINGS
    def test_injection_attempts_blocked(self, admin_only_policy, user, doc):
        """INVARIANT: Injection fields should not grant admin access."""
        from ragguard.policy.engine import PolicyEngine

        engine = PolicyEngine(admin_only_policy)

        # User should only get access if they have "admin" in their actual roles
        actual_roles = user.get("roles", [])
        if actual_roles is None:
            actual_roles = []
        elif isinstance(actual_roles, str):
            actual_roles = [actual_roles]

        has_admin = "admin" in actual_roles
        result = engine.evaluate(user, doc)

        if has_admin:
            assert result is True, "Admin user was denied"
        else:
            assert result is False, f"Non-admin user was granted access via injection: {user}"

    @given(user=user_context(), doc=document_metadata())
    @HYPOTHESIS_SETTINGS
    def test_evaluation_is_deterministic(self, admin_only_policy, user, doc):
        """INVARIANT: Same inputs should always produce same output."""
        from ragguard.policy.engine import PolicyEngine

        engine = PolicyEngine(admin_only_policy)

        # Evaluate multiple times
        results = [engine.evaluate(user, doc) for _ in range(5)]

        # All results should be identical
        assert all(r == results[0] for r in results), \
            f"Non-deterministic evaluation: {results}"

    @given(user=user_context(), doc=document_metadata())
    @HYPOTHESIS_SETTINGS
    def test_evaluation_never_crashes(self, admin_only_policy, user, doc):
        """INVARIANT: Evaluation should never crash, always return bool."""
        from ragguard.policy.engine import PolicyEngine

        engine = PolicyEngine(admin_only_policy)

        # Should not raise any exception
        result = engine.evaluate(user, doc)

        assert isinstance(result, bool), f"Result should be bool, got {type(result)}"

    @given(user=user_context())
    @HYPOTHESIS_SETTINGS
    def test_filter_generation_never_crashes(self, admin_only_policy, user):
        """INVARIANT: Filter generation should never crash unexpectedly."""
        from ragguard.policy.engine import PolicyEngine

        engine = PolicyEngine(admin_only_policy)

        for backend in ["qdrant", "chromadb", "pinecone", "milvus"]:
            try:
                result = engine.to_filter(user, backend)
                # Result can be None, dict, or backend-specific object (e.g. Qdrant Filter)
                # Just verify it returns something without crashing
                assert result is None or result is not None  # Always true, just checking no crash
            except ValueError as e:
                # Validation errors for invalid roles are expected security behavior
                error_msg = str(e).lower()
                assert any(keyword in error_msg for keyword in [
                    "invalid", "role", "empty", "character"
                ]), f"Unexpected ValueError: {e}"
            except Exception as e:
                # Only acceptable exceptions are for unsupported backends
                error_msg = str(e).lower()
                assert "backend" in error_msg or "not supported" in error_msg, \
                    f"Unexpected exception for {backend}: {e}"


class TestConditionEvaluationProperties:
    """Property-based tests for condition evaluation."""

    @pytest.fixture
    def equality_policy(self):
        """Policy with equality condition."""
        from ragguard.policy.models import Policy
        return Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "equality_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.department == document.department"]
                    }
                }
            ],
            "default": "deny"
        })

    @given(dept=st.text(min_size=1, max_size=50))
    @HYPOTHESIS_SETTINGS
    def test_equality_reflexive(self, equality_policy, dept):
        """INVARIANT: If dept matches, access should be granted."""
        from ragguard.policy.engine import PolicyEngine

        # Skip whitespace-only or problematic values
        assume(dept.strip() == dept and len(dept.strip()) > 0)

        engine = PolicyEngine(equality_policy)
        user = {"id": "test", "roles": ["user"], "department": dept}
        doc = {"id": "doc1", "department": dept}

        result = engine.evaluate(user, doc)
        assert result is True, f"Same department '{dept}' should grant access"

    @given(dept1=st.text(min_size=1, max_size=50), dept2=st.text(min_size=1, max_size=50))
    @HYPOTHESIS_SETTINGS
    def test_equality_distinct(self, equality_policy, dept1, dept2):
        """INVARIANT: Different departments should deny access."""
        from ragguard.policy.engine import PolicyEngine

        # Skip if they're actually equal
        assume(dept1 != dept2)
        assume(dept1.strip() == dept1 and dept2.strip() == dept2)
        assume(len(dept1.strip()) > 0 and len(dept2.strip()) > 0)

        engine = PolicyEngine(equality_policy)
        user = {"id": "test", "roles": ["user"], "department": dept1}
        doc = {"id": "doc1", "department": dept2}

        result = engine.evaluate(user, doc)
        assert result is False, f"Different departments '{dept1}' vs '{dept2}' should deny"

    @given(user_val=st.one_of(st.none(), st.text(max_size=10)),
           doc_val=st.one_of(st.none(), st.text(max_size=10)))
    @HYPOTHESIS_SETTINGS
    def test_none_handling_security(self, equality_policy, user_val, doc_val):
        """INVARIANT: None values should not inadvertently grant access."""
        from ragguard.policy.engine import PolicyEngine

        engine = PolicyEngine(equality_policy)
        user = {"id": "test", "roles": ["user"]}
        doc = {"id": "doc1"}

        if user_val is not None:
            user["department"] = user_val
        if doc_val is not None:
            doc["department"] = doc_val

        result = engine.evaluate(user, doc)

        # Access should only be granted if both have the same non-None, non-empty value
        both_have_value = user_val is not None and doc_val is not None
        values_equal = user_val == doc_val
        values_non_empty = bool(user_val and doc_val)

        if both_have_value and values_equal and values_non_empty:
            # May or may not grant - depends on exact values
            pass
        elif user_val is None or doc_val is None:
            # At least one is None - should deny
            assert result is False, \
                f"None value should not grant access: user={user_val}, doc={doc_val}"


class TestRoleHandlingProperties:
    """Property-based tests for role handling."""

    @pytest.fixture
    def multi_role_policy(self):
        """Policy with multiple required roles."""
        from ragguard.policy.models import Policy
        return Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "admin_rule",
                    "allow": {"roles": ["admin", "superuser"]}
                },
                {
                    "name": "user_rule",
                    "allow": {"roles": ["user"]}
                }
            ],
            "default": "deny"
        })

    @given(roles=st.lists(st.text(min_size=1, max_size=20), max_size=10))
    @HYPOTHESIS_SETTINGS
    def test_role_membership_determines_access(self, multi_role_policy, roles):
        """INVARIANT: Access determined by role membership."""
        from ragguard.policy.engine import PolicyEngine

        engine = PolicyEngine(multi_role_policy)
        user = {"id": "test", "roles": roles}
        doc = {"id": "doc1"}

        result = engine.evaluate(user, doc)

        # Should grant if any allowed role is present
        allowed_roles = {"admin", "superuser", "user"}
        has_allowed_role = bool(set(roles) & allowed_roles)

        assert result == has_allowed_role, \
            f"Roles {roles} should {'grant' if has_allowed_role else 'deny'} access"

    @given(base_roles=st.lists(st.text(min_size=1, max_size=20), max_size=5),
           extra_fields=st.dictionaries(
               st.text(min_size=1, max_size=20),
               st.one_of(st.text(max_size=20), st.lists(st.text(max_size=10), max_size=5)),
               max_size=5
           ))
    @HYPOTHESIS_SETTINGS
    def test_extra_fields_do_not_affect_role_check(self, multi_role_policy, base_roles, extra_fields):
        """INVARIANT: Extra user fields should not affect role-based access."""
        from ragguard.policy.engine import PolicyEngine

        engine = PolicyEngine(multi_role_policy)

        # Create user with just roles
        user_basic = {"id": "test", "roles": base_roles}

        # Create user with extra fields
        user_extra = {"id": "test", "roles": base_roles, **extra_fields}
        # Ensure we don't override roles
        user_extra["roles"] = base_roles

        doc = {"id": "doc1"}

        result_basic = engine.evaluate(user_basic, doc)
        result_extra = engine.evaluate(user_extra, doc)

        assert result_basic == result_extra, \
            f"Extra fields changed result: basic={result_basic}, extra={result_extra}"


class TestFilterGenerationProperties:
    """Property-based tests for filter generation."""

    @pytest.fixture
    def standard_policy(self):
        """Standard policy for filter tests."""
        from ragguard.policy.models import Policy
        return Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "dept_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.department == document.department"]
                    }
                },
                {
                    "name": "public_rule",
                    "match": {"visibility": "public"},
                    "allow": {"everyone": True}
                }
            ],
            "default": "deny"
        })

    @given(user=user_context())
    @HYPOTHESIS_SETTINGS
    def test_qdrant_filter_structure(self, standard_policy, user):
        """INVARIANT: Qdrant filters should have valid structure."""
        from ragguard.policy.engine import PolicyEngine

        engine = PolicyEngine(standard_policy)

        try:
            result = engine.to_filter(user, "qdrant")
            # Result can be None or a Qdrant Filter object (or dict for some backends)
            # Just verify it returns without crashing
            if result is not None:
                # Qdrant returns Filter objects, check it has expected attributes
                assert hasattr(result, 'must') or hasattr(result, 'should') or isinstance(result, dict), \
                    f"Unexpected Qdrant filter type: {type(result)}"
        except ValueError as e:
            # Validation errors for invalid roles are expected security behavior
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in [
                "invalid", "role", "empty", "character"
            ]), f"Unexpected ValueError: {e}"
        except Exception as e:
            # Should not crash unexpectedly
            pytest.fail(f"Filter generation crashed: {e}")

    @given(user=user_context())
    @HYPOTHESIS_SETTINGS
    def test_chromadb_filter_structure(self, standard_policy, user):
        """INVARIANT: ChromaDB filters should have valid structure."""
        from ragguard.policy.engine import PolicyEngine

        engine = PolicyEngine(standard_policy)

        try:
            result = engine.to_filter(user, "chromadb")

            if result is not None:
                assert isinstance(result, dict), "ChromaDB filter should be dict"
                # Valid ChromaDB operators
                valid_operators = {"$and", "$or", "$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"}
                # Structure varies but should be dict
        except ValueError as e:
            # Validation errors for invalid roles are expected security behavior
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in [
                "invalid", "role", "empty", "character"
            ]), f"Unexpected ValueError: {e}"
        except Exception as e:
            pytest.fail(f"Filter generation crashed: {e}")


class TestExplainerProperties:
    """Property-based tests for policy explainer."""

    @pytest.fixture
    def complex_policy(self):
        """Complex policy for explainer tests."""
        from ragguard.policy.models import Policy
        return Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "admin_override",
                    "allow": {"roles": ["admin"]}
                },
                {
                    "name": "dept_access",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.department == document.department"]
                    }
                },
                {
                    "name": "public_access",
                    "match": {"visibility": "public"},
                    "allow": {"everyone": True}
                }
            ],
            "default": "deny"
        })

    @given(user=user_context(), doc=document_metadata())
    @HYPOTHESIS_SETTINGS
    def test_explainer_matches_evaluation(self, complex_policy, user, doc):
        """INVARIANT: Explainer decision should match engine evaluation."""
        from ragguard.policy.engine import PolicyEngine
        from ragguard.policy.explainer import QueryExplainer

        engine = PolicyEngine(complex_policy)
        explainer = QueryExplainer(complex_policy)

        eval_result = engine.evaluate(user, doc)
        explain_result = explainer.explain(user, doc)

        expected_decision = "ALLOW" if eval_result else "DENY"
        assert explain_result.final_decision == expected_decision, \
            f"Explainer decision {explain_result.final_decision} != engine result {eval_result}"

    @given(user=user_context(), doc=document_metadata())
    @HYPOTHESIS_SETTINGS
    def test_explainer_never_crashes(self, complex_policy, user, doc):
        """INVARIANT: Explainer should never crash."""
        from ragguard.policy.explainer import QueryExplainer

        explainer = QueryExplainer(complex_policy)

        # Should not raise
        result = explainer.explain(user, doc)

        # Should have expected attributes
        assert hasattr(result, 'final_decision')
        assert hasattr(result, 'rule_evaluations')
        assert result.final_decision in ("ALLOW", "DENY")


class TestEdgeCaseInputs:
    """Property-based tests for edge case inputs."""

    @pytest.fixture
    def basic_policy(self):
        """Basic policy for edge case tests."""
        from ragguard.policy.models import Policy
        return Policy.from_dict({
            "version": "1",
            "rules": [
                {"name": "user_rule", "allow": {"roles": ["user"]}}
            ],
            "default": "deny"
        })

    @given(special_char=st.sampled_from([
        '\x00', '\n', '\r', '\t', '\x1b', '\xff',
        '\\', '"', "'", '`', '$', '{', '}',
        '<', '>', '&', '|', ';', '\u0000'
    ]))
    @HYPOTHESIS_SETTINGS
    def test_special_characters_handled(self, basic_policy, special_char):
        """INVARIANT: Special characters should not cause crashes or bypasses."""
        from ragguard.policy.engine import PolicyEngine

        engine = PolicyEngine(basic_policy)

        # Test in user ID
        user = {"id": f"user{special_char}test", "roles": ["user"]}
        doc = {"id": f"doc{special_char}test"}

        result = engine.evaluate(user, doc)
        assert isinstance(result, bool)

        # Test in role
        user = {"id": "test", "roles": [f"user{special_char}"]}
        result = engine.evaluate(user, doc)
        assert isinstance(result, bool)

    @given(size=st.integers(min_value=0, max_value=1000))
    @HYPOTHESIS_SETTINGS
    def test_large_role_lists(self, basic_policy, size):
        """INVARIANT: Large role lists should be handled without crash."""
        from ragguard.policy.engine import PolicyEngine

        engine = PolicyEngine(basic_policy)

        roles = [f"role_{i}" for i in range(size)]
        user = {"id": "test", "roles": roles}
        doc = {"id": "doc1"}

        result = engine.evaluate(user, doc)

        # Should deny unless "user" is in roles
        expected = "user" in roles
        assert result == expected

    @given(depth=st.integers(min_value=1, max_value=10))
    @HYPOTHESIS_SETTINGS
    def test_nested_user_data(self, basic_policy, depth):
        """INVARIANT: Nested user data should not cause issues."""
        from ragguard.policy.engine import PolicyEngine

        engine = PolicyEngine(basic_policy)

        # Build nested structure
        nested = {"value": "nested"}
        for _ in range(depth):
            nested = {"nested": nested}

        user = {"id": "test", "roles": ["user"], "metadata": nested}
        doc = {"id": "doc1"}

        result = engine.evaluate(user, doc)
        assert result is True  # Has user role
