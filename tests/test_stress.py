"""
Stress tests for RAGGuard to validate performance under load and edge cases.

These tests verify:
1. DoS protection limits are enforced
2. Performance with large rule sets
3. Memory usage with complex policies
4. Concurrent access patterns
5. Edge cases with extreme values
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from pydantic import ValidationError

# Skip all tests if qdrant-client is not installed (required for filter generation tests)
pytest.importorskip("qdrant_client")

from ragguard import Policy, PolicyParser
from ragguard.exceptions import PolicyValidationError
from ragguard.policy import PolicyEngine
from ragguard.policy.models import PolicyLimits


class TestPolicyLimits:
    """Test that policy complexity limits are enforced."""

    def test_max_rules_limit(self):
        """Test that policies with too many rules are rejected."""
        # Create a policy with MAX_RULES + 1 rules
        rules = [
            {"name": f"rule-{i}", "allow": {"everyone": True}}
            for i in range(PolicyLimits.MAX_RULES + 1)
        ]

        with pytest.raises((ValidationError, ValueError), match="Too many rules"):
            Policy.from_dict({
                "version": "1",
                "rules": rules,
                "default": "deny"
            }, validate=False)

    def test_max_conditions_per_rule_limit(self):
        """Test that rules with too many conditions are rejected."""
        conditions = [
            f"user.field{i} == document.field{i}"
            for i in range(PolicyLimits.MAX_CONDITIONS_PER_RULE + 1)
        ]

        with pytest.raises((ValidationError, ValueError), match="Too many conditions"):
            Policy.from_dict({
                "version": "1",
                "rules": [{
                    "name": "overloaded-rule",
                    "allow": {"conditions": conditions}
                }],
                "default": "deny"
            }, validate=False)

    def test_max_total_conditions_limit(self):
        """Test that total conditions across all rules are limited."""
        # Create rules that together exceed MAX_TOTAL_CONDITIONS
        num_rules = 20
        conditions_per_rule = (PolicyLimits.MAX_TOTAL_CONDITIONS // num_rules) + 1

        rules = []
        for i in range(num_rules):
            conditions = [
                f"user.field{j} == document.field{j}"
                for j in range(conditions_per_rule)
            ]
            rules.append({
                "name": f"rule-{i}",
                "allow": {"conditions": conditions}
            })

        with pytest.raises((ValidationError, ValueError), match="Too many total conditions"):
            Policy.from_dict({
                "version": "1",
                "rules": rules,
                "default": "deny"
            }, validate=False)

    def test_max_list_size_limit(self):
        """Test that large list literals in conditions are rejected."""
        # Create a list literal that exceeds MAX_LIST_SIZE
        large_list = ", ".join([f"'item{i}'" for i in range(PolicyLimits.MAX_LIST_SIZE + 1)])

        with pytest.raises((ValidationError, ValueError), match="List literal too large"):
            Policy.from_dict({
                "version": "1",
                "rules": [{
                    "name": "large-list",
                    "allow": {
                        "conditions": [f"user.role in [{large_list}]"]
                    }
                }],
                "default": "deny"
            }, validate=False)

    def test_max_list_size_bytes_limit(self):
        """Test that list literals exceeding byte size are rejected."""
        # Create a list with a single huge element
        huge_element = "x" * (PolicyLimits.MAX_LIST_SIZE_BYTES + 1)

        with pytest.raises((ValidationError, ValueError), match="List literal too large"):
            Policy.from_dict({
                "version": "1",
                "rules": [{
                    "name": "huge-element",
                    "allow": {
                        "conditions": [f"user.role in ['{huge_element}']"]
                    }
                }],
                "default": "deny"
            }, validate=False)

    def test_max_policy_size_bytes_limit(self):
        """Test that huge policy files are rejected."""
        # Create a policy that exceeds MAX_POLICY_SIZE_BYTES
        # Use very long rule names to inflate size
        huge_name = "x" * 100000  # 100KB per rule

        rules = []
        for i in range(15):  # 15 * 100KB = 1.5MB
            rules.append({
                "name": f"{huge_name}-{i}",
                "allow": {"everyone": True}
            })

        with pytest.raises(ValueError, match="Policy too large"):
            Policy.from_dict({
                "version": "1",
                "rules": rules,
                "default": "deny"
            }, validate=False)

    def test_max_nesting_depth_limit(self):
        """Test that deeply nested match conditions are rejected."""
        # Create a deeply nested dict
        nested = {}
        current = nested
        for i in range(PolicyLimits.MAX_NESTING_DEPTH + 1):
            current[f"level{i}"] = {}
            current = current[f"level{i}"]
        current["value"] = "deep"

        with pytest.raises((ValidationError, ValueError), match="nesting too deep"):
            Policy.from_dict({
                "version": "1",
                "rules": [{
                    "name": "deep-nesting",
                    "match": nested,
                    "allow": {"everyone": True}
                }],
                "default": "deny"
            }, validate=False)


class TestPerformance:
    """Test performance with realistic loads."""

    def test_large_rule_set_performance(self):
        """Test evaluation performance with maximum allowed rules."""
        # Create a policy with MAX_RULES rules
        rules = []
        for i in range(PolicyLimits.MAX_RULES):
            rules.append({
                "name": f"rule-{i}",
                "match": {"category": f"cat-{i}"},
                "allow": {"roles": [f"role-{i}"]}
            })

        policy = Policy.from_dict({
            "version": "1",
            "rules": rules,
            "default": "deny"
        }, validate=False)

        engine = PolicyEngine(policy)
        user = {"id": "alice", "roles": ["role-50"]}
        doc = {"id": "doc1", "category": "cat-50"}

        # Measure evaluation time
        start = time.time()
        for _ in range(100):
            engine.evaluate(user, doc)
        elapsed = time.time() - start

        # Should complete 100 evaluations in under 1 second
        assert elapsed < 1.0, f"Performance regression: {elapsed:.2f}s for 100 evals"

    def test_complex_conditions_performance(self):
        """Test performance with complex condition evaluation."""
        # Create rule with many conditions
        conditions = [
            f"user.attr{i} == document.attr{i}"
            for i in range(50)
        ]

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "complex",
                "allow": {"conditions": conditions}
            }],
            "default": "deny"
        }, validate=False)

        engine = PolicyEngine(policy)

        # Create matching user and document
        user = {"id": "alice"}
        doc = {"id": "doc1"}
        for i in range(50):
            user[f"attr{i}"] = f"value{i}"
            doc[f"attr{i}"] = f"value{i}"

        # Measure evaluation time
        start = time.time()
        for _ in range(1000):
            engine.evaluate(user, doc)
        elapsed = time.time() - start

        # Should complete 1000 evaluations in under 1 second
        assert elapsed < 1.0, f"Performance regression: {elapsed:.2f}s for 1000 evals"

    def test_concurrent_policy_evaluation(self):
        """Test thread-safe concurrent policy evaluation."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "department-access",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            }],
            "default": "deny"
        }, validate=False)

        engine = PolicyEngine(policy)

        def evaluate_for_user(user_id: int) -> bool:
            user = {"id": f"user-{user_id}", "department": "engineering"}
            doc = {"id": "doc1", "department": "engineering"}
            return engine.evaluate(user, doc)

        # Run 100 concurrent evaluations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(evaluate_for_user, i) for i in range(100)]

            results = [f.result() for f in as_completed(futures)]

        # All should succeed
        assert all(results), "Some concurrent evaluations failed"
        assert len(results) == 100

    def test_filter_generation_performance(self):
        """Test filter generation performance for different backends."""
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
                        "conditions": ["user.department == document.department"]
                    }
                }
            ],
            "default": "deny"
        }, validate=False)

        engine = PolicyEngine(policy)
        user = {"id": "alice", "department": "engineering"}

        backends = ["qdrant", "pgvector", "weaviate", "pinecone", "chromadb"]

        for backend in backends:
            start = time.time()
            for _ in range(1000):
                engine.to_filter(user, backend)
            elapsed = time.time() - start

            # Should generate 1000 filters in under 0.5 seconds per backend
            assert elapsed < 0.5, f"{backend} filter generation too slow: {elapsed:.2f}s"


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_empty_user_context(self):
        """Test evaluation with minimal user context."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "public",
                "match": {"visibility": "public"},
                "allow": {"everyone": True}
            }],
            "default": "deny"
        }, validate=False)

        engine = PolicyEngine(policy)

        # User with only ID
        user = {"id": "alice"}
        doc = {"id": "doc1", "visibility": "public"}

        assert engine.evaluate(user, doc) == True

    def test_very_long_field_names(self):
        """Test handling of very long field names."""
        long_field = "x" * 1000

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "long-field",
                "allow": {
                    "conditions": [f"user.{long_field} == document.{long_field}"]
                }
            }],
            "default": "deny"
        }, validate=False)

        engine = PolicyEngine(policy)
        user = {"id": "alice", long_field: "value"}
        doc = {"id": "doc1", long_field: "value"}

        assert engine.evaluate(user, doc) == True

    def test_very_long_field_values(self):
        """Test handling of very long field values."""
        long_value = "x" * 10000

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "long-value",
                "allow": {
                    "conditions": ["user.token == document.token"]
                }
            }],
            "default": "deny"
        }, validate=False)

        engine = PolicyEngine(policy)
        user = {"id": "alice", "token": long_value}
        doc = {"id": "doc1", "token": long_value}

        assert engine.evaluate(user, doc) == True

    def test_unicode_field_names_and_values(self):
        """Test handling of Unicode in field names and values."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "unicode",
                "allow": {
                    "conditions": ["user.部門 == document.部門"]
                }
            }],
            "default": "deny"
        }, validate=False)

        engine = PolicyEngine(policy)
        user = {"id": "alice", "部門": "エンジニアリング"}
        doc = {"id": "doc1", "部門": "エンジニアリング"}

        assert engine.evaluate(user, doc) == True

    def test_special_characters_in_values(self):
        """Test handling of special characters in values."""
        special_chars = "!@#$%^&*()[]{}|\\:;\"'<>,.?/~`"

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "special",
                "allow": {
                    "conditions": ["user.code == document.code"]
                }
            }],
            "default": "deny"
        }, validate=False)

        engine = PolicyEngine(policy)
        user = {"id": "alice", "code": special_chars}
        doc = {"id": "doc1", "code": special_chars}

        assert engine.evaluate(user, doc) == True

    def test_numeric_field_edge_cases(self):
        """Test edge cases with numeric values."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "numeric",
                "allow": {
                    "conditions": [
                        "user.score == document.min_score"
                    ]
                }
            }],
            "default": "deny"
        }, validate=False)

        engine = PolicyEngine(policy)

        # Test very large numbers
        user = {"id": "alice", "score": 999999999999999}
        doc = {"id": "doc1", "min_score": 999999999999999}
        assert engine.evaluate(user, doc) == True

        # Test very small numbers
        user = {"id": "alice", "score": 0.00000000001}
        doc = {"id": "doc1", "min_score": 0.00000000001}
        assert engine.evaluate(user, doc) == True

        # Test negative numbers
        user = {"id": "alice", "score": -999999}
        doc = {"id": "doc1", "min_score": -999999}
        assert engine.evaluate(user, doc) == True

    def test_deeply_nested_field_access(self):
        """Test deeply nested field access in conditions."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "nested",
                "allow": {
                    "conditions": [
                        "user.meta.team.department.name == document.meta.team.department.name"
                    ]
                }
            }],
            "default": "deny"
        }, validate=False)

        engine = PolicyEngine(policy)
        user = {
            "id": "alice",
            "meta": {
                "team": {
                    "department": {
                        "name": "engineering"
                    }
                }
            }
        }
        doc = {
            "id": "doc1",
            "meta": {
                "team": {
                    "department": {
                        "name": "engineering"
                    }
                }
            }
        }

        assert engine.evaluate(user, doc) == True

    def test_list_with_many_elements(self):
        """Test list matching with many elements."""
        # Create a list with 100 roles
        roles = [f"role-{i}" for i in range(100)]

        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "many-roles",
                "allow": {"roles": roles}
            }],
            "default": "deny"
        }, validate=False)

        engine = PolicyEngine(policy)
        user = {"id": "alice", "roles": ["role-50"]}
        doc = {"id": "doc1"}

        assert engine.evaluate(user, doc) == True

    def test_missing_fields_at_various_depths(self):
        """Test handling of missing fields at different nesting levels."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "nested-check",
                "allow": {
                    "conditions": [
                        "user.a.b.c == document.x.y.z"
                    ]
                }
            }],
            "default": "deny"
        }, validate=False)

        engine = PolicyEngine(policy)

        # Missing at first level
        user = {"id": "alice"}
        doc = {"id": "doc1"}
        assert engine.evaluate(user, doc) == False

        # Missing at second level
        user = {"id": "alice", "a": {}}
        doc = {"id": "doc1", "x": {}}
        assert engine.evaluate(user, doc) == False

        # Missing at third level
        user = {"id": "alice", "a": {"b": {}}}
        doc = {"id": "doc1", "x": {"y": {}}}
        assert engine.evaluate(user, doc) == False


class TestMemoryUsage:
    """Test memory efficiency."""

    def test_policy_reuse(self):
        """Test that PolicyEngine can be reused for many evaluations."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "dept-access",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            }],
            "default": "deny"
        }, validate=False)

        engine = PolicyEngine(policy)

        # Perform many evaluations to test memory stability
        for i in range(10000):
            user = {"id": f"user-{i}", "department": "engineering"}
            doc = {"id": f"doc-{i}", "department": "engineering"}
            result = engine.evaluate(user, doc)
            assert result == True

    def test_filter_generation_reuse(self):
        """Test that filter generation doesn't leak memory."""
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "dept-access",
                "allow": {
                    "conditions": ["user.department == document.department"]
                }
            }],
            "default": "deny"
        }, validate=False)

        engine = PolicyEngine(policy)

        # Generate many filters
        for i in range(10000):
            user = {"id": f"user-{i}", "department": f"dept-{i % 10}"}
            filter_obj = engine.to_filter(user, "qdrant")
            assert filter_obj is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
