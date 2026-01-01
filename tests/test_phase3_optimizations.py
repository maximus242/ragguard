"""
Tests for Phase 3 optimizations: import optimization and role sets.

Verifies that:
- Filter builders are pre-loaded at module level
- Role lists are converted to sets for O(1) lookup
- All optimizations maintain correctness
"""

import pytest

# Skip all tests if qdrant-client is not installed (required for filter builder tests)
pytest.importorskip("qdrant_client")

from ragguard.policy import Policy, PolicyEngine
from ragguard.policy.engine import _FILTER_BUILDERS

# ============================================================================
# Filter Builder Registry Tests
# ============================================================================

def test_filter_builders_preloaded():
    """Test that filter builders are pre-loaded in registry."""
    # Verify all expected backends are in registry
    expected_backends = ["qdrant", "pgvector", "weaviate", "pinecone", "chromadb", "faiss"]
    for backend in expected_backends:
        assert backend in _FILTER_BUILDERS

    # Verify Qdrant builder is callable
    assert callable(_FILTER_BUILDERS["qdrant"])
    assert callable(_FILTER_BUILDERS["pgvector"])
    assert callable(_FILTER_BUILDERS["weaviate"])
    assert callable(_FILTER_BUILDERS["pinecone"])
    assert callable(_FILTER_BUILDERS["chromadb"])

    # FAISS should be None (no native filtering)
    assert _FILTER_BUILDERS["faiss"] is None


def test_filter_builder_registry_lookup():
    """Test that filter building uses registry for fast lookup."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "admin", "allow": {"roles": ["admin"]}}
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy, enable_filter_cache=False)
    user = {"roles": ["admin"]}

    # Should work without errors using registry
    filter_qdrant = engine.to_filter(user, "qdrant")
    filter_pgvector = engine.to_filter(user, "pgvector")

    # Filters should be built
    assert filter_qdrant is not None
    assert filter_pgvector is not None


def test_unsupported_backend_error():
    """Test that unsupported backends raise appropriate error."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "admin", "allow": {"roles": ["admin"]}}
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy)
    user = {"roles": ["admin"]}

    with pytest.raises(Exception, match="Unsupported Backend"):
        engine.to_filter(user, "unsupported_db")


# ============================================================================
# Role Set Optimization Tests
# ============================================================================

def test_role_sets_preconverted():
    """Test that role lists are pre-converted to sets."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "admin-rule",
                "allow": {"roles": ["admin", "superuser", "moderator"]}
            },
            {
                "name": "user-rule",
                "allow": {"roles": ["user"]}
            }
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Verify role sets were created
    assert 0 in engine._role_sets
    assert 1 in engine._role_sets

    # Verify they are sets, not lists
    assert isinstance(engine._role_sets[0], set)
    assert isinstance(engine._role_sets[1], set)

    # Verify content
    assert engine._role_sets[0] == {"admin", "superuser", "moderator"}
    assert engine._role_sets[1] == {"user"}


def test_role_set_evaluation_correctness():
    """Test that role set evaluation produces correct results."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "privileged",
                "allow": {"roles": ["admin", "superuser", "moderator"]}
            }
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy)
    doc = {"text": "secret"}

    # Admin should be allowed
    admin_user = {"roles": ["admin"]}
    assert engine.evaluate(admin_user, doc) is True

    # Moderator should be allowed
    moderator_user = {"roles": ["moderator"]}
    assert engine.evaluate(moderator_user, doc) is True

    # Regular user should be denied
    regular_user = {"roles": ["user"]}
    assert engine.evaluate(regular_user, doc) is False

    # User with multiple roles, one matching
    multi_role_user = {"roles": ["user", "guest", "admin"]}
    assert engine.evaluate(multi_role_user, doc) is True


def test_role_set_with_single_role_string():
    """Test role set handling when user has single role as string."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "admin", "allow": {"roles": ["admin"]}}
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy)
    doc = {"text": "secret"}

    # Single role as string should work
    user = {"roles": "admin"}
    assert engine.evaluate(user, doc) is True


def test_role_set_with_none_roles():
    """Test role set handling when user has None roles."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "admin", "allow": {"roles": ["admin"]}}
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy)
    doc = {"text": "secret"}

    # None roles should be treated as empty list
    user = {"roles": None}
    assert engine.evaluate(user, doc) is False


def test_role_set_with_missing_roles():
    """Test role set handling when user has no roles field."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "admin", "allow": {"roles": ["admin"]}}
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy)
    doc = {"text": "secret"}

    # Missing roles field should be treated as empty list
    user = {"id": "user123"}
    assert engine.evaluate(user, doc) is False


def test_rule_without_roles():
    """Test that rules without roles don't create role sets."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "condition-only",
                "allow": {
                    "conditions": ["user.dept == document.dept"]
                }
            }
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # No role sets should be created for this rule
    assert 0 not in engine._role_sets or len(engine._role_sets[0]) == 0


# ============================================================================
# Combined Optimization Tests
# ============================================================================

def test_all_optimizations_together():
    """Test that all Phase 3 optimizations work together correctly."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "dept-docs",
                "match": {"type": "internal"},
                "allow": {
                    "roles": ["employee", "contractor"],
                    "conditions": ["user.department == document.department"]
                }
            }
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy, enable_filter_cache=True)

    # Verify role set created
    assert 0 in engine._role_sets
    assert engine._role_sets[0] == {"employee", "contractor"}

    # Test evaluation
    user = {"roles": ["employee"], "department": "engineering"}
    doc_match = {"type": "internal", "department": "engineering"}
    doc_nomatch = {"type": "internal", "department": "sales"}

    assert engine.evaluate(user, doc_match) is True
    assert engine.evaluate(user, doc_nomatch) is False

    # Test filter generation (uses registry)
    filter_obj = engine.to_filter(user, "qdrant")
    assert filter_obj is not None

    # Verify caching works
    stats = engine.get_cache_stats()
    assert stats is not None
    assert stats["size"] == 1  # One filter cached


def test_performance_with_many_roles():
    """Test performance benefit with many roles (set lookup vs list iteration)."""
    # Create policy with many roles
    many_roles = [f"role_{i}" for i in range(100)]

    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {"name": "many-roles", "allow": {"roles": many_roles}}
        ],
        "default": "deny"
    })

    engine = PolicyEngine(policy)

    # Verify set was created
    assert 0 in engine._role_sets
    assert len(engine._role_sets[0]) == 100

    # Test with role at end of list (worst case for list iteration)
    user = {"roles": ["role_99"]}
    doc = {"text": "test"}

    # Should still be fast with set lookup
    import time
    iterations = 1000

    start = time.time()
    for _ in range(iterations):
        engine.evaluate(user, doc)
    elapsed = time.time() - start

    # Should complete 1000 iterations quickly (< 100ms)
    assert elapsed < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
