#!/usr/bin/env python3
"""
Comprehensive Weakness Analysis for RAGGuard v0.2.0

This script identifies potential issues, edge cases, and gaps in the current implementation.
"""

from ragguard import Policy
from ragguard.policy.engine import PolicyEngine
from ragguard.filters.builder import (
    to_qdrant_filter,
    to_pgvector_filter,
    to_weaviate_filter,
    to_pinecone_filter,
    to_chromadb_filter
)
import time

print("=" * 80)
print("RAGGuard v0.2.0 - Comprehensive Weakness Analysis")
print("=" * 80)

issues_found = []
tests_passed = 0
tests_failed = 0

def test(name, func):
    """Run a test and track results."""
    global tests_passed, tests_failed, issues_found
    print(f"\n🔍 {name}")
    try:
        result, issue = func()
        if result:
            print(f"   ✅ PASS")
            tests_passed += 1
        else:
            print(f"   ⚠️  POTENTIAL ISSUE: {issue}")
            issues_found.append((name, issue))
            tests_failed += 1
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)[:100]}")
        issues_found.append((name, f"Exception: {str(e)[:100]}"))
        tests_failed += 1

# ============================================================================
# 1. LIST LITERAL EDGE CASES
# ============================================================================

def test_nested_lists():
    """Test nested lists in conditions."""
    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "nested",
                "allow": {
                    "everyone": True,
                    "conditions": ["document.tags in [['a', 'b'], 'c']"]
                }
            }],
            "default": "deny"
        })
        engine = PolicyEngine(policy)
        doc = {"tags": ["a", "b"]}
        result = engine.evaluate({}, doc)
        return False, "Nested lists parsed but may not work correctly"
    except Exception as e:
        return True, None  # Good - should reject nested lists

test("Nested lists in list literals", test_nested_lists)

def test_list_with_quotes():
    """Test list literals with quotes inside strings."""
    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "quotes",
                "allow": {
                    "everyone": True,
                    "conditions": ["document.author in [\"O'Reilly\", \"Bob's\"]"]
                }
            }],
            "default": "deny"
        })
        engine = PolicyEngine(policy)
        doc = {"author": "O'Reilly"}
        result = engine.evaluate({}, doc)

        if result:
            return True, None
        else:
            return False, "Quotes in list literals not handled correctly"
    except Exception as e:
        return False, f"Failed to parse list with quotes: {e}"

test("List literals with internal quotes", test_list_with_quotes)

def test_very_large_list():
    """Test list with 1000+ items."""
    large_list = [f"item_{i}" for i in range(1000)]
    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "large",
                "allow": {
                    "everyone": True,
                    "conditions": [f"document.id in {large_list}"]
                }
            }],
            "default": "deny"
        })
        engine = PolicyEngine(policy)
        doc = {"id": "item_500"}

        start = time.time()
        result = engine.evaluate({}, doc)
        elapsed = time.time() - start

        if elapsed > 0.1:  # 100ms threshold
            return False, f"Large list evaluation slow: {elapsed*1000:.1f}ms"

        return True, None
    except Exception as e:
        return False, f"Failed with large list: {e}"

test("Very large list (1000 items)", test_very_large_list)

def test_empty_string_in_list():
    """Test empty string in list literal."""
    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "empty",
                "allow": {
                    "everyone": True,
                    "conditions": ["document.status in ['active', '', 'pending']"]
                }
            }],
            "default": "deny"
        })
        engine = PolicyEngine(policy)
        doc = {"status": ""}
        result = engine.evaluate({}, doc)

        if result:
            return True, None
        else:
            return False, "Empty string in list not matched"
    except Exception as e:
        return False, f"Failed with empty string: {e}"

test("Empty string in list literal", test_empty_string_in_list)

def test_special_chars_in_list():
    """Test special characters in list literals."""
    special_values = ["@admin", "#dev", "$prod", "test&qa", "a|b", "a;b"]
    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "special",
                "allow": {
                    "everyone": True,
                    "conditions": [f"document.tag in {special_values}"]
                }
            }],
            "default": "deny"
        })
        engine = PolicyEngine(policy)

        all_match = True
        for val in special_values:
            doc = {"tag": val}
            if not engine.evaluate({}, doc):
                all_match = False
                return False, f"Special char value '{val}' not matched"

        return True, None
    except Exception as e:
        return False, f"Failed with special chars: {e}"

test("Special characters in list literals", test_special_chars_in_list)

# ============================================================================
# 2. CROSS-BACKEND CONSISTENCY
# ============================================================================

def test_cross_backend_filter_consistency():
    """Test that all backends generate consistent filters."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "test",
            "allow": {
                "everyone": True,
                "conditions": [
                    "document.category in ['cs.AI', 'cs.LG']",
                    "document.status != 'archived'"
                ]
            }
        }],
        "default": "deny"
    })

    user = {"id": "test"}

    # Generate filters for all backends
    try:
        qdrant_filter = to_qdrant_filter(policy, user)
        pgvector_filter = to_pgvector_filter(policy, user)
        weaviate_filter = to_weaviate_filter(policy, user)
        pinecone_filter = to_pinecone_filter(policy, user)
        chromadb_filter = to_chromadb_filter(policy, user)

        # All should be non-None
        if None in [qdrant_filter, pgvector_filter, weaviate_filter, pinecone_filter, chromadb_filter]:
            return False, "Some backends returned None filter"

        # Can't easily verify equivalence, but at least they all generated something
        return True, None
    except Exception as e:
        return False, f"Cross-backend filter generation failed: {e}"

test("Cross-backend filter consistency", test_cross_backend_filter_consistency)

# ============================================================================
# 3. ERROR MESSAGE QUALITY
# ============================================================================

def test_malformed_list_error():
    """Test error message for malformed list literal."""
    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "bad",
                "allow": {
                    "everyone": True,
                    "conditions": ["document.field in ['a', 'b'"]  # Missing closing ]
                }
            }],
            "default": "deny"
        })
        return False, "Malformed list accepted without clear error"
    except Exception as e:
        error_msg = str(e)
        if "list" in error_msg.lower() or "bracket" in error_msg.lower():
            return True, None
        else:
            return False, f"Error message unclear: {error_msg}"

test("Error message for malformed list", test_malformed_list_error)

def test_invalid_operator_combo_error():
    """Test error for invalid operator combinations."""
    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "bad",
                "allow": {
                    "everyone": True,
                    "conditions": ["document.field !== 'value'"]  # Invalid operator
                }
            }],
            "default": "deny"
        })
        engine = PolicyEngine(policy)
        result = engine.evaluate({}, {"field": "value"})
        return False, "Invalid operator !== accepted"
    except Exception as e:
        return True, None

test("Error message for invalid operator", test_invalid_operator_combo_error)

# ============================================================================
# 4. PERFORMANCE CONCERNS
# ============================================================================

def test_multiple_not_in_performance():
    """Test performance with multiple NOT IN conditions."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "multi-not-in",
            "allow": {
                "everyone": True,
                "conditions": [
                    "document.status not in ['a', 'b', 'c', 'd', 'e']",
                    "document.type not in ['x', 'y', 'z']",
                    "document.region not in ['r1', 'r2', 'r3', 'r4']"
                ]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy)
    doc = {"status": "active", "type": "doc", "region": "us"}

    # Time 1000 evaluations
    start = time.time()
    for _ in range(1000):
        engine.evaluate({}, doc)
    elapsed = time.time() - start

    per_eval = elapsed / 1000
    if per_eval > 0.001:  # 1ms threshold
        return False, f"Multiple NOT IN slow: {per_eval*1000:.2f}ms per eval"

    return True, None

test("Performance with multiple NOT IN", test_multiple_not_in_performance)

def test_filter_cache_effectiveness():
    """Test that filter cache is actually being used."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "test",
            "allow": {
                "everyone": True,
                "conditions": ["document.category in ['cs.AI', 'cs.LG']"]
            }
        }],
        "default": "deny"
    })

    engine = PolicyEngine(policy, backend="qdrant")
    user = {"id": "alice"}

    # First call - cache miss
    start1 = time.time()
    filter1 = engine.to_filter(user, "qdrant")
    time1 = time.time() - start1

    # Second call - should be cached
    start2 = time.time()
    filter2 = engine.to_filter(user, "qdrant")
    time2 = time.time() - start2

    # Check cache stats
    stats = engine.get_cache_stats()
    if stats is None:
        return False, "Cache stats not available"

    if stats['hits'] == 0:
        return False, "Cache not being used (0 hits)"

    if time2 > time1 * 0.5:  # Cached should be much faster
        return False, f"Cache not providing speedup: {time1*1e6:.1f}µs vs {time2*1e6:.1f}µs"

    return True, None

test("Filter cache effectiveness", test_filter_cache_effectiveness)

# ============================================================================
# 5. EDGE CASES IN POLICY SEMANTICS
# ============================================================================

def test_empty_conditions_list():
    """Test behavior with empty conditions array."""
    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "empty",
                "allow": {
                    "everyone": True,
                    "conditions": []  # Empty array
                }
            }],
            "default": "deny"
        })
        engine = PolicyEngine(policy)
        result = engine.evaluate({}, {"field": "value"})

        # Should allow (everyone=true, no conditions to fail)
        if result:
            return True, None
        else:
            return False, "Empty conditions list denies access"
    except Exception as e:
        return False, f"Failed with empty conditions: {e}"

test("Empty conditions array", test_empty_conditions_list)

def test_null_in_list():
    """Test None/null in list literal."""
    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "null",
                "allow": {
                    "everyone": True,
                    "conditions": ["document.field in ['a', None, 'b']"]
                }
            }],
            "default": "deny"
        })
        engine = PolicyEngine(policy)
        doc = {"field": None}
        result = engine.evaluate({}, doc)

        # Should it match? This is a design decision
        return True, None  # Accept either behavior for now
    except Exception as e:
        return False, f"Failed with None in list: {e}"

test("None/null in list literal", test_null_in_list)

def test_mixed_type_list():
    """Test list with mixed types."""
    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "mixed",
                "allow": {
                    "everyone": True,
                    "conditions": ["document.field in ['string', 123, true]"]
                }
            }],
            "default": "deny"
        })
        engine = PolicyEngine(policy)

        # Test string
        if not engine.evaluate({}, {"field": "string"}):
            return False, "String not matched in mixed-type list"

        # Test number
        if not engine.evaluate({}, {"field": 123}):
            return False, "Number not matched in mixed-type list"

        # Test boolean
        if not engine.evaluate({}, {"field": True}):
            return False, "Boolean not matched in mixed-type list"

        return True, None
    except Exception as e:
        return False, f"Failed with mixed-type list: {e}"

test("Mixed-type list literal", test_mixed_type_list)

# ============================================================================
# 6. BACKEND-SPECIFIC LIMITATIONS
# ============================================================================

def test_pgvector_max_params():
    """Test pgvector with many parameters (SQL param limit)."""
    # PostgreSQL has a limit on the number of parameters
    large_list = [f"val_{i}" for i in range(500)]
    try:
        policy = Policy.from_dict({
            "version": "1",
            "rules": [{
                "name": "large",
                "allow": {
                    "everyone": True,
                    "conditions": [f"document.id in {large_list}"]
                }
            }],
            "default": "deny"
        })

        sql, params = to_pgvector_filter(policy, {})

        if len(params) > 500:
            return False, f"pgvector generated {len(params)} params (may exceed DB limit)"

        return True, None
    except Exception as e:
        return False, f"pgvector failed with large list: {e}"

test("pgvector max parameters", test_pgvector_max_params)

# ============================================================================
# 7. DOCUMENTATION GAPS
# ============================================================================

def test_docstring_coverage():
    """Check if key functions have docstrings."""
    from ragguard.filters import builder

    functions_to_check = [
        '_parse_list_literal',
        'to_qdrant_filter',
        'to_pgvector_filter',
    ]

    missing = []
    for func_name in functions_to_check:
        if hasattr(builder, func_name):
            func = getattr(builder, func_name)
            if func.__doc__ is None or len(func.__doc__.strip()) < 10:
                missing.append(func_name)

    if missing:
        return False, f"Missing docstrings: {', '.join(missing)}"

    return True, None

test("Docstring coverage", test_docstring_coverage)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("WEAKNESS ANALYSIS SUMMARY")
print("=" * 80)
print(f"\n✅ Tests Passed: {tests_passed}")
print(f"⚠️  Issues Found: {tests_failed}")

if issues_found:
    print("\n" + "=" * 80)
    print("ISSUES DETECTED")
    print("=" * 80)
    for i, (name, issue) in enumerate(issues_found, 1):
        print(f"\n{i}. {name}")
        print(f"   Issue: {issue}")

print("\n" + "=" * 80)
print("ADDITIONAL WEAKNESSES NOT TESTED")
print("=" * 80)

weaknesses = [
    ("Integration Testing", "Only Qdrant and Weaviate have end-to-end tests"),
    ("Load Testing", "Only tested with 100 concurrent users, not 1000+"),
    ("Fuzzing", "No fuzzing tests for malformed inputs"),
    ("Type Safety", "No mypy type checking enabled"),
    ("Code Coverage", "No code coverage metrics tracked"),
    ("Comparison Operators", "Missing >, <, >=, <= operators"),
    ("Regex Support", "No regex pattern matching"),
    ("String Operations", "No contains/startswith/endswith"),
    ("Date Comparisons", "No time-based access control"),
    ("Metrics/Monitoring", "No Prometheus metrics or distributed tracing"),
    ("Health Checks", "No /health endpoint for k8s readiness"),
    ("Structured Logging", "Logs not structured (JSON) for log aggregation"),
    ("Multi-Backend", "No testing with multiple backends simultaneously"),
    ("Cache Tuning", "No documentation on cache size tuning for production"),
    ("Memory Profiling", "No memory usage profiling with large policies"),
    ("Query Complexity", "No limits on filter complexity to prevent DoS"),
    ("Rate Limiting", "No rate limiting on policy evaluation"),
    ("Audit Compliance", "No GDPR/HIPAA compliance documentation"),
]

for i, (area, weakness) in enumerate(weaknesses, 1):
    print(f"\n{i}. {area}")
    print(f"   {weakness}")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

recommendations = [
    ("Priority 1", [
        "Add integration tests for pgvector, Pinecone, ChromaDB, FAISS",
        "Add cross-backend consistency tests",
        "Add fuzzing tests for condition parsing",
        "Document backend-specific limitations",
    ]),
    ("Priority 2", [
        "Add mypy type checking to CI/CD",
        "Add code coverage tracking (target: 95%+)",
        "Add load testing with 1000+ concurrent users",
        "Add performance regression tests",
    ]),
    ("Priority 3", [
        "Implement comparison operators (>, <, >=, <=)",
        "Add Prometheus metrics",
        "Add structured logging",
        "Add health check endpoint",
    ]),
]

for priority, items in recommendations:
    print(f"\n{priority}:")
    for item in items:
        print(f"  - {item}")

print("\n" + "=" * 80)
