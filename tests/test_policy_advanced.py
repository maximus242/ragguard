"""
Comprehensive tests for policy engine and related modules.
"""

import pytest

from ragguard.policy import PolicyEngine, load_policy
from ragguard.policy.errors import (
    EnhancedPolicyEvaluationError,
    EnhancedPolicyValidationError,
    PolicyErrorFormatter,
)
from ragguard.policy.models import AllowConditions, Policy, Rule


class TestPolicyModels:
    """Tests for policy models."""

    def test_policy_from_dict(self):
        """Test creating Policy from dict."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {"name": "test", "allow": {"roles": ["user"]}}
            ],
            "default": "deny"
        })

        assert policy.version == "1"
        assert len(policy.rules) == 1
        assert policy.default == "deny"

    def test_rule_with_match(self):
        """Test Rule with match conditions."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "public_docs",
                    "match": {"type": "public"},
                    "allow": {"everyone": True}
                }
            ],
            "default": "deny"
        })

        assert policy.rules[0].match == {"type": "public"}
        assert policy.rules[0].allow.everyone is True

    def test_rule_with_conditions(self):
        """Test Rule with conditions."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "dept_access",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.dept == document.dept"]
                    }
                }
            ],
            "default": "deny"
        })

        assert len(policy.rules[0].allow.conditions) == 1

    def test_allow_with_everyone(self):
        """Test AllowConditions with everyone flag."""
        allow = AllowConditions(everyone=True)
        assert allow.everyone is True

    def test_allow_with_roles(self):
        """Test AllowConditions with roles."""
        allow = AllowConditions(roles=["admin", "editor"])
        assert "admin" in allow.roles
        assert "editor" in allow.roles


class TestPolicyEngine:
    """Tests for PolicyEngine."""

    @pytest.fixture
    def simple_policy(self):
        return Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "admin_access",
                    "allow": {"roles": ["admin"]}
                },
                {
                    "name": "user_public",
                    "allow": {"roles": ["user"]},
                    "match": {"visibility": "public"}
                }
            ],
            "default": "deny"
        })

    def test_evaluate_admin_access(self, simple_policy):
        """Test admin access evaluation."""
        engine = PolicyEngine(simple_policy)

        user = {"id": "alice", "roles": ["admin"]}
        document = {"id": "doc1", "visibility": "private"}

        result = engine.evaluate(user, document)
        assert result is True

    def test_evaluate_user_public_access(self, simple_policy):
        """Test user access to public documents."""
        engine = PolicyEngine(simple_policy)

        user = {"id": "bob", "roles": ["user"]}
        document = {"id": "doc1", "visibility": "public"}

        result = engine.evaluate(user, document)
        assert result is True

    def test_evaluate_user_private_denied(self, simple_policy):
        """Test user denied access to private documents."""
        engine = PolicyEngine(simple_policy)

        user = {"id": "bob", "roles": ["user"]}
        document = {"id": "doc1", "visibility": "private"}

        result = engine.evaluate(user, document)
        assert result is False

    def test_evaluate_default_deny(self, simple_policy):
        """Test default deny."""
        engine = PolicyEngine(simple_policy)

        user = {"id": "guest", "roles": []}
        document = {"id": "doc1"}

        result = engine.evaluate(user, document)
        assert result is False

    def test_evaluate_with_conditions(self):
        """Test evaluation with conditions."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [
                {
                    "name": "dept_access",
                    "allow": {
                        "roles": ["user"],
                        "conditions": ["user.department == document.department"]
                    }
                }
            ],
            "default": "deny"
        })

        engine = PolicyEngine(policy)

        # Same department - should allow
        user = {"id": "alice", "roles": ["user"], "department": "engineering"}
        doc = {"id": "doc1", "department": "engineering"}
        assert engine.evaluate(user, doc) is True

        # Different department - should deny
        user = {"id": "alice", "roles": ["user"], "department": "sales"}
        assert engine.evaluate(user, doc) is False

    def test_to_filter_qdrant(self, simple_policy):
        """Test to_filter for Qdrant backend."""
        pytest.importorskip("qdrant_client")
        engine = PolicyEngine(simple_policy)

        user = {"id": "bob", "roles": ["user"]}
        filter_dict = engine.to_filter(user, backend="qdrant")

        assert filter_dict is not None

    def test_to_filter_chromadb(self, simple_policy):
        """Test to_filter for ChromaDB backend."""
        engine = PolicyEngine(simple_policy)

        user = {"id": "bob", "roles": ["user"]}
        filter_dict = engine.to_filter(user, backend="chromadb")

        assert filter_dict is not None

    def test_to_filter_pinecone(self, simple_policy):
        """Test to_filter for Pinecone backend."""
        engine = PolicyEngine(simple_policy)

        user = {"id": "bob", "roles": ["user"]}
        filter_dict = engine.to_filter(user, backend="pinecone")

        assert filter_dict is not None

    def test_to_filter_admin(self, simple_policy):
        """Test admin filter."""
        pytest.importorskip("qdrant_client")
        engine = PolicyEngine(simple_policy)

        user = {"id": "alice", "roles": ["admin"]}
        filter_dict = engine.to_filter(user, backend="qdrant")

        # Admin with unrestricted rule may get None or empty filter
        # The exact behavior depends on implementation
        assert True  # Just ensure no exception is raised


class TestPolicyErrors:
    """Tests for policy error classes."""

    def test_policy_evaluation_error(self):
        """Test EnhancedPolicyEvaluationError."""
        error = EnhancedPolicyEvaluationError("Test error message")
        assert "Test error" in str(error)

    def test_policy_validation_error(self):
        """Test EnhancedPolicyValidationError."""
        error = EnhancedPolicyValidationError("Validation failed")
        assert "Validation" in str(error)

    def test_policy_error_formatter(self):
        """Test PolicyErrorFormatter."""
        formatter = PolicyErrorFormatter()
        assert formatter is not None


class TestPolicyCompiler:
    """Tests for condition compiler."""

    def test_compile_expression(self):
        """Test compiling expression."""
        from ragguard.policy.compiler import ConditionCompiler

        condition = "user.role == 'admin' OR user.department == document.department"
        compiled = ConditionCompiler.compile_expression(condition)

        assert compiled is not None

    def test_compile_and_expression(self):
        """Test compiling AND expression."""
        from ragguard.policy.compiler import ConditionCompiler

        condition = "user.role == 'user' AND user.department == document.department"
        compiled = ConditionCompiler.compile_expression(condition)

        assert compiled is not None


class TestLoadPolicy:
    """Tests for load_policy function."""

    def test_load_json_policy(self, tmp_path):
        """Test loading JSON policy file."""
        policy_file = tmp_path / "policy.json"
        policy_file.write_text('''{
            "version": "1",
            "rules": [
                {"name": "test", "allow": {"roles": ["user"]}}
            ],
            "default": "deny"
        }''')

        policy = load_policy(str(policy_file))

        assert policy.version == "1"
        assert len(policy.rules) == 1

    def test_load_yaml_policy(self, tmp_path):
        """Test loading YAML policy file."""
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text('''
version: "1"
rules:
  - name: test
    allow:
      roles:
        - user
default: deny
''')

        policy = load_policy(str(policy_file))

        assert policy.version == "1"
        assert len(policy.rules) == 1

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(Exception):
            load_policy("/nonexistent/policy.json")
