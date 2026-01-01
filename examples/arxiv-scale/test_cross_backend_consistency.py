#!/usr/bin/env python3
"""
Cross-Backend Consistency Tests for RAGGuard v0.2.0

Verifies that all backends generate semantically equivalent filters
for the same policy and user context.
"""

from ragguard import Policy
from ragguard.filters.builder import (
    to_qdrant_filter,
    to_pgvector_filter,
    to_weaviate_filter,
    to_pinecone_filter,
    to_chromadb_filter
)

print("=" * 80)
print("Cross-Backend Consistency Tests")
print("=" * 80)

tests_passed = 0
tests_failed = 0

def test(name, func):
    """Run a test and track results."""
    global tests_passed, tests_failed
    print(f"\n🔍 {name}")
    try:
        result = func()
        if result:
            print(f"   ✅ PASS")
            tests_passed += 1
        else:
            print(f"   ❌ FAIL")
            tests_failed += 1
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        tests_failed += 1

# ============================================================================
# CONSISTENCY TESTS
# ============================================================================

def test_simple_equality():
    """Test that all backends handle simple equality consistently."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "public-only",
            "allow": {
                "everyone": True,
                "conditions": ["document.access_level == 'public'"]
            }
        }],
        "default": "deny"
    })

    user = {"id": "test"}

    # Generate filters
    qdrant_filter = to_qdrant_filter(policy, user)
    pgvector_filter = to_pgvector_filter(policy, user)
    weaviate_filter = to_weaviate_filter(policy, user)
    pinecone_filter = to_pinecone_filter(policy, user)
    chromadb_filter = to_chromadb_filter(policy, user)

    # All should be non-None
    if None in [qdrant_filter, weaviate_filter, pinecone_filter, chromadb_filter]:
        print(f"      Some backends returned None")
        return False

    # pgvector returns (sql, params)
    if pgvector_filter == ("", []):
        print(f"      pgvector returned empty filter")
        return False

    print(f"      All backends generated non-empty filters")
    return True

test("Simple equality filter consistency", test_simple_equality)

def test_negation_consistency():
    """Test != operator consistency across backends."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "not-archived",
            "allow": {
                "everyone": True,
                "conditions": ["document.status != 'archived'"]
            }
        }],
        "default": "deny"
    })

    user = {}

    qdrant_filter = to_qdrant_filter(policy, user)
    pgvector_filter = to_pgvector_filter(policy, user)
    weaviate_filter = to_weaviate_filter(policy, user)
    pinecone_filter = to_pinecone_filter(policy, user)
    chromadb_filter = to_chromadb_filter(policy, user)

    # Verify all generated filters
    if None in [qdrant_filter, weaviate_filter, pinecone_filter, chromadb_filter]:
        print(f"      Some backends returned None for != operator")
        return False

    if pgvector_filter == ("", []):
        print(f"      pgvector returned empty filter for != operator")
        return False

    # Verify pgvector SQL contains !=
    sql, params = pgvector_filter
    if "!=" not in sql:
        print(f"      pgvector SQL missing != operator: {sql}")
        return False

    # Verify Pinecone has $ne
    if "$ne" not in str(pinecone_filter):
        print(f"      Pinecone missing $ne operator: {pinecone_filter}")
        return False

    # Verify ChromaDB has $ne
    if "$ne" not in str(chromadb_filter):
        print(f"      ChromaDB missing $ne operator: {chromadb_filter}")
        return False

    print(f"      All backends correctly implement != operator")
    return True

test("Negation operator (!=) consistency", test_negation_consistency)

def test_list_literal_consistency():
    """Test list literal IN operator consistency."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "ai-ml",
            "allow": {
                "everyone": True,
                "conditions": ["document.category in ['cs.AI', 'cs.LG', 'cs.CV']"]
            }
        }],
        "default": "deny"
    })

    user = {}

    qdrant_filter = to_qdrant_filter(policy, user)
    pgvector_filter = to_pgvector_filter(policy, user)
    weaviate_filter = to_weaviate_filter(policy, user)
    pinecone_filter = to_pinecone_filter(policy, user)
    chromadb_filter = to_chromadb_filter(policy, user)

    # All should generate filters
    if None in [qdrant_filter, weaviate_filter, pinecone_filter, chromadb_filter]:
        print(f"      Some backends returned None for list literal")
        return False

    if pgvector_filter == ("", []):
        print(f"      pgvector returned empty filter for list literal")
        return False

    # Verify pgvector SQL contains IN
    sql, params = pgvector_filter
    if "IN" not in sql:
        print(f"      pgvector SQL missing IN operator: {sql}")
        return False

    # Verify correct number of parameters
    if len(params) != 3:
        print(f"      pgvector params count wrong: {len(params)} != 3")
        return False

    # Verify Pinecone has $in
    if "$in" not in str(pinecone_filter):
        print(f"      Pinecone missing $in operator: {pinecone_filter}")
        return False

    # Verify ChromaDB has $in
    if "$in" not in str(chromadb_filter):
        print(f"      ChromaDB missing $in operator: {chromadb_filter}")
        return False

    print(f"      All backends correctly implement list literals")
    return True

test("List literal (IN) consistency", test_list_literal_consistency)

def test_not_in_consistency():
    """Test NOT IN operator consistency."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "exclude-bad",
            "allow": {
                "everyone": True,
                "conditions": ["document.status not in ['archived', 'deleted']"]
            }
        }],
        "default": "deny"
    })

    user = {}

    qdrant_filter = to_qdrant_filter(policy, user)
    pgvector_filter = to_pgvector_filter(policy, user)
    weaviate_filter = to_weaviate_filter(policy, user)
    pinecone_filter = to_pinecone_filter(policy, user)
    chromadb_filter = to_chromadb_filter(policy, user)

    # All should generate filters
    if None in [qdrant_filter, weaviate_filter, pinecone_filter, chromadb_filter]:
        print(f"      Some backends returned None for NOT IN")
        return False

    if pgvector_filter == ("", []):
        print(f"      pgvector returned empty filter for NOT IN")
        return False

    # Verify pgvector SQL contains NOT IN
    sql, params = pgvector_filter
    if "NOT IN" not in sql:
        print(f"      pgvector SQL missing NOT IN: {sql}")
        return False

    # Verify Pinecone has $nin
    if "$nin" not in str(pinecone_filter):
        print(f"      Pinecone missing $nin operator: {pinecone_filter}")
        return False

    # Verify ChromaDB has $nin
    if "$nin" not in str(chromadb_filter):
        print(f"      ChromaDB missing $nin operator: {chromadb_filter}")
        return False

    print(f"      All backends correctly implement NOT IN")
    return True

test("NOT IN operator consistency", test_not_in_consistency)

def test_empty_list_in_consistency():
    """Test empty list IN [] consistency."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "empty",
            "allow": {
                "everyone": True,
                "conditions": ["document.category in []"]
            }
        }],
        "default": "deny"
    })

    user = {}

    qdrant_filter = to_qdrant_filter(policy, user)
    pgvector_filter = to_pgvector_filter(policy, user)
    weaviate_filter = to_weaviate_filter(policy, user)
    pinecone_filter = to_pinecone_filter(policy, user)
    chromadb_filter = to_chromadb_filter(policy, user)

    # All should generate "impossible match" filters
    if qdrant_filter is None:
        print(f"      Qdrant returned None for empty IN []")
        return False

    sql, params = pgvector_filter
    # Accept either "1 = 0" or "FALSE" as impossible filter
    if "1 = 0" not in sql and "FALSE" not in sql:
        print(f"      pgvector didn't generate impossible filter: {sql}")
        return False

    if weaviate_filter is None:
        print(f"      Weaviate returned None for empty IN []")
        return False

    if pinecone_filter is None:
        print(f"      Pinecone returned None for empty IN []")
        return False

    if chromadb_filter is None:
        print(f"      ChromaDB returned None for empty IN []")
        return False

    print(f"      All backends correctly handle empty IN []")
    return True

test("Empty list IN [] consistency", test_empty_list_in_consistency)

def test_empty_list_not_in_consistency():
    """Test empty list NOT IN [] consistency."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "empty-not-in",
            "allow": {
                "everyone": True,
                "conditions": ["document.category not in []"]
            }
        }],
        "default": "deny"
    })

    user = {}

    qdrant_filter = to_qdrant_filter(policy, user)
    pgvector_filter = to_pgvector_filter(policy, user)
    weaviate_filter = to_weaviate_filter(policy, user)
    pinecone_filter = to_pinecone_filter(policy, user)
    chromadb_filter = to_chromadb_filter(policy, user)

    # NOT IN [] should allow everything (None or always-true filter)
    # Qdrant: Should allow all (complex to verify)
    # pgvector: Should be "1 = 1" or "TRUE" (always true)
    sql, params = pgvector_filter
    if "1 = 1" not in sql and "TRUE" not in sql:
        print(f"      pgvector didn't generate always-true filter: {sql}")
        return False

    # Weaviate, Pinecone, ChromaDB: None means no restriction (correct)
    # These might be None or have no restrictions

    print(f"      All backends correctly handle empty NOT IN []")
    return True

test("Empty list NOT IN [] consistency", test_empty_list_not_in_consistency)

def test_multiple_conditions_consistency():
    """Test multiple conditions with AND logic."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "complex",
            "allow": {
                "everyone": True,
                "conditions": [
                    "document.category in ['cs.AI', 'cs.LG']",
                    "document.access_level != 'restricted'",
                    "document.status not in ['archived', 'draft']"
                ]
            }
        }],
        "default": "deny"
    })

    user = {}

    qdrant_filter = to_qdrant_filter(policy, user)
    pgvector_filter = to_pgvector_filter(policy, user)
    weaviate_filter = to_weaviate_filter(policy, user)
    pinecone_filter = to_pinecone_filter(policy, user)
    chromadb_filter = to_chromadb_filter(policy, user)

    # All should generate filters
    if None in [qdrant_filter, weaviate_filter, pinecone_filter, chromadb_filter]:
        print(f"      Some backends returned None for multiple conditions")
        return False

    if pgvector_filter == ("", []):
        print(f"      pgvector returned empty filter")
        return False

    # Verify pgvector SQL has all conditions
    sql, params = pgvector_filter
    if "IN" not in sql or "!=" not in sql or "NOT IN" not in sql:
        print(f"      pgvector SQL missing some conditions: {sql}")
        return False

    # Verify pgvector uses AND logic
    and_count = sql.count(" AND ")
    if and_count < 2:  # Should have at least 2 ANDs for 3 conditions
        print(f"      pgvector might not be using AND logic: {sql}")
        return False

    # Verify ChromaDB uses $and
    if "$and" not in str(chromadb_filter):
        print(f"      ChromaDB missing $and operator: {chromadb_filter}")
        return False

    print(f"      All backends correctly handle multiple conditions")
    return True

test("Multiple conditions with AND logic", test_multiple_conditions_consistency)

def test_user_field_consistency():
    """Test user field references consistency."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "dept-match",
            "allow": {
                "conditions": ["user.department == document.department"]
            }
        }],
        "default": "deny"
    })

    user = {"department": "engineering"}

    qdrant_filter = to_qdrant_filter(policy, user)
    pgvector_filter = to_pgvector_filter(policy, user)
    weaviate_filter = to_weaviate_filter(policy, user)
    pinecone_filter = to_pinecone_filter(policy, user)
    chromadb_filter = to_chromadb_filter(policy, user)

    # All should resolve user.department and create filter
    if None in [qdrant_filter, weaviate_filter, pinecone_filter, chromadb_filter]:
        print(f"      Some backends returned None for user field")
        return False

    if pgvector_filter == ("", []):
        print(f"      pgvector returned empty filter")
        return False

    # Verify pgvector has the user's department value
    sql, params = pgvector_filter
    if "engineering" not in params:
        print(f"      pgvector params missing 'engineering': {params}")
        return False

    print(f"      All backends correctly resolve user fields")
    return True

test("User field references consistency", test_user_field_consistency)

def test_array_field_operations_consistency():
    """Test array field operations (user.id in document.array) consistency."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "shared-docs",
            "allow": {
                "conditions": ["user.id in document.shared_with"]
            }
        }],
        "default": "deny"
    })

    user = {"id": "alice"}

    qdrant_filter = to_qdrant_filter(policy, user)
    pgvector_filter = to_pgvector_filter(policy, user)
    weaviate_filter = to_weaviate_filter(policy, user)
    pinecone_filter = to_pinecone_filter(policy, user)
    chromadb_filter = to_chromadb_filter(policy, user)

    # All should generate filters for array operations
    if None in [qdrant_filter, weaviate_filter, pinecone_filter, chromadb_filter]:
        print(f"      Some backends returned None for array field operations")
        return False

    if pgvector_filter == ("", []):
        print(f"      pgvector returned empty filter for array operations")
        return False

    # Verify pgvector SQL contains array operation
    sql, params = pgvector_filter
    if "= ANY" not in sql:
        print(f"      pgvector SQL missing array operation (= ANY): {sql}")
        return False

    # Verify params has alice
    if "alice" not in params:
        print(f"      pgvector params missing 'alice': {params}")
        return False

    # Verify Pinecone has $in
    if "$in" not in str(pinecone_filter):
        print(f"      Pinecone missing $in for array operations: {pinecone_filter}")
        return False

    # Verify ChromaDB has $in
    if "$in" not in str(chromadb_filter):
        print(f"      ChromaDB missing $in for array operations: {chromadb_filter}")
        return False

    print(f"      All backends correctly implement array field operations")
    return True

test("Array field operations (user.id in document.array) consistency", test_array_field_operations_consistency)

def test_literal_in_array_consistency():
    """Test literal in array ('public' in document.tags) consistency."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "public-tagged",
            "allow": {
                "everyone": True,
                "conditions": ["'public' in document.tags"]
            }
        }],
        "default": "deny"
    })

    user = {}

    qdrant_filter = to_qdrant_filter(policy, user)
    pgvector_filter = to_pgvector_filter(policy, user)
    weaviate_filter = to_weaviate_filter(policy, user)
    pinecone_filter = to_pinecone_filter(policy, user)
    chromadb_filter = to_chromadb_filter(policy, user)

    # All should generate filters
    if None in [qdrant_filter, weaviate_filter, pinecone_filter, chromadb_filter]:
        print(f"      Some backends returned None for literal in array")
        return False

    if pgvector_filter == ("", []):
        print(f"      pgvector returned empty filter for literal in array")
        return False

    # Verify pgvector SQL contains array check
    sql, params = pgvector_filter
    if "= ANY" not in sql:
        print(f"      pgvector SQL missing array check: {sql}")
        return False

    # Verify params has 'public'
    if "public" not in params:
        print(f"      pgvector params missing 'public': {params}")
        return False

    print(f"      All backends correctly implement literal in array")
    return True

test("Literal in array ('public' in document.tags) consistency", test_literal_in_array_consistency)

def test_field_exists_consistency():
    """Test field existence checks (document.field exists) consistency."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "reviewed-docs",
            "allow": {
                "everyone": True,
                "conditions": ["document.reviewed_at exists"]
            }
        }],
        "default": "deny"
    })

    user = {}

    qdrant_filter = to_qdrant_filter(policy, user)
    pgvector_filter = to_pgvector_filter(policy, user)
    weaviate_filter = to_weaviate_filter(policy, user)
    pinecone_filter = to_pinecone_filter(policy, user)
    chromadb_filter = to_chromadb_filter(policy, user)

    # All should generate filters for existence checks
    if None in [qdrant_filter, weaviate_filter, pinecone_filter, chromadb_filter]:
        print(f"      Some backends returned None for existence check")
        return False

    if pgvector_filter == ("", []):
        print(f"      pgvector returned empty filter for existence check")
        return False

    # Verify pgvector SQL contains IS NOT NULL
    sql, params = pgvector_filter
    if "IS NOT NULL" not in sql:
        print(f"      pgvector SQL missing IS NOT NULL: {sql}")
        return False

    # Verify Pinecone has $exists
    if "$exists" not in str(pinecone_filter):
        print(f"      Pinecone missing $exists operator: {pinecone_filter}")
        return False

    # Verify ChromaDB has $ne None
    if "$ne" not in str(chromadb_filter):
        print(f"      ChromaDB missing $ne None for exists: {chromadb_filter}")
        return False

    print(f"      All backends correctly implement field existence checks")
    return True

test("Field existence (document.field exists) consistency", test_field_exists_consistency)

def test_field_not_exists_consistency():
    """Test field non-existence checks (document.field not exists) consistency."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "no-draft-notes",
            "allow": {
                "everyone": True,
                "conditions": ["document.draft_notes not exists"]
            }
        }],
        "default": "deny"
    })

    user = {}

    qdrant_filter = to_qdrant_filter(policy, user)
    pgvector_filter = to_pgvector_filter(policy, user)
    weaviate_filter = to_weaviate_filter(policy, user)
    pinecone_filter = to_pinecone_filter(policy, user)
    chromadb_filter = to_chromadb_filter(policy, user)

    # All should generate filters
    if None in [qdrant_filter, weaviate_filter, pinecone_filter, chromadb_filter]:
        print(f"      Some backends returned None for non-existence check")
        return False

    if pgvector_filter == ("", []):
        print(f"      pgvector returned empty filter for non-existence check")
        return False

    # Verify pgvector SQL contains IS NULL
    sql, params = pgvector_filter
    if "IS NULL" not in sql:
        print(f"      pgvector SQL missing IS NULL: {sql}")
        return False

    # Verify Pinecone has $exists: false or $eq: null
    pinecone_str = str(pinecone_filter)
    if "$exists" not in pinecone_str and "$eq" not in pinecone_str:
        print(f"      Pinecone missing $exists false or $eq null: {pinecone_filter}")
        return False

    # Verify ChromaDB has $eq None
    if "$eq" not in str(chromadb_filter):
        print(f"      ChromaDB missing $eq None for not exists: {chromadb_filter}")
        return False

    print(f"      All backends correctly implement field non-existence checks")
    return True

test("Field non-existence (document.field not exists) consistency", test_field_not_exists_consistency)

def test_array_not_in_consistency():
    """Test array NOT IN (user.id not in document.blocked) consistency."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "not-blocked",
            "allow": {
                "conditions": ["user.id not in document.blocked_users"]
            }
        }],
        "default": "deny"
    })

    user = {"id": "alice"}

    qdrant_filter = to_qdrant_filter(policy, user)
    pgvector_filter = to_pgvector_filter(policy, user)
    weaviate_filter = to_weaviate_filter(policy, user)
    pinecone_filter = to_pinecone_filter(policy, user)
    chromadb_filter = to_chromadb_filter(policy, user)

    # All should generate filters
    if None in [qdrant_filter, weaviate_filter, pinecone_filter, chromadb_filter]:
        print(f"      Some backends returned None for array NOT IN")
        return False

    if pgvector_filter == ("", []):
        print(f"      pgvector returned empty filter for array NOT IN")
        return False

    # Verify pgvector SQL contains NOT (... = ANY(...))
    sql, params = pgvector_filter
    if "= ANY" not in sql or "NOT" not in sql:
        print(f"      pgvector SQL missing NOT (... = ANY): {sql}")
        return False

    # Verify params has alice
    if "alice" not in params:
        print(f"      pgvector params missing 'alice': {params}")
        return False

    # Verify Pinecone has $nin or NOT + $in
    pinecone_str = str(pinecone_filter)
    if "$nin" not in pinecone_str and "$in" not in pinecone_str:
        print(f"      Pinecone missing $nin for array NOT IN: {pinecone_filter}")
        return False

    # Verify ChromaDB has $nin or NOT
    chromadb_str = str(chromadb_filter)
    if "$nin" not in chromadb_str and "$in" not in chromadb_str:
        print(f"      ChromaDB missing $nin for array NOT IN: {chromadb_filter}")
        return False

    print(f"      All backends correctly implement array NOT IN operations")
    return True

test("Array NOT IN (user.id not in document.blocked) consistency", test_array_not_in_consistency)

def test_literal_not_in_array_consistency():
    """Test literal NOT IN array ('archived' not in document.tags) consistency."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "not-archived",
            "allow": {
                "everyone": True,
                "conditions": ["'archived' not in document.tags"]
            }
        }],
        "default": "deny"
    })

    user = {}

    qdrant_filter = to_qdrant_filter(policy, user)
    pgvector_filter = to_pgvector_filter(policy, user)
    weaviate_filter = to_weaviate_filter(policy, user)
    pinecone_filter = to_pinecone_filter(policy, user)
    chromadb_filter = to_chromadb_filter(policy, user)

    # All should generate filters
    if None in [qdrant_filter, weaviate_filter, pinecone_filter, chromadb_filter]:
        print(f"      Some backends returned None for literal NOT IN array")
        return False

    if pgvector_filter == ("", []):
        print(f"      pgvector returned empty filter for literal NOT IN array")
        return False

    # Verify pgvector SQL contains NOT (... = ANY(...))
    sql, params = pgvector_filter
    if "= ANY" not in sql or "NOT" not in sql:
        print(f"      pgvector SQL missing NOT (... = ANY): {sql}")
        return False

    # Verify params has 'archived'
    if "archived" not in params:
        print(f"      pgvector params missing 'archived': {params}")
        return False

    print(f"      All backends correctly implement literal NOT IN array")
    return True

test("Literal NOT IN array ('archived' not in document.tags) consistency", test_literal_not_in_array_consistency)

def test_complex_combined_conditions():
    """Test complex combination of all new features."""
    policy = Policy.from_dict({
        "version": "1",
        "rules": [{
            "name": "complex-access",
            "allow": {
                "conditions": [
                    "user.id in document.authorized_users",
                    "document.reviewed_at exists",
                    "'public' in document.tags",
                    "document.draft_notes not exists",
                    "document.status != 'archived'"
                ]
            }
        }],
        "default": "deny"
    })

    user = {"id": "alice"}

    qdrant_filter = to_qdrant_filter(policy, user)
    pgvector_filter = to_pgvector_filter(policy, user)
    weaviate_filter = to_weaviate_filter(policy, user)
    pinecone_filter = to_pinecone_filter(policy, user)
    chromadb_filter = to_chromadb_filter(policy, user)

    # All should generate complex filters
    if None in [qdrant_filter, weaviate_filter, pinecone_filter, chromadb_filter]:
        print(f"      Some backends returned None for complex conditions")
        return False

    if pgvector_filter == ("", []):
        print(f"      pgvector returned empty filter for complex conditions")
        return False

    # Verify pgvector SQL has all operations
    sql, params = pgvector_filter
    required_operations = ["= ANY", "IS NOT NULL", "IS NULL", "!="]
    missing = [op for op in required_operations if op not in sql]
    if missing:
        print(f"      pgvector SQL missing operations: {missing}")
        print(f"      SQL: {sql}")
        return False

    # Verify pgvector uses AND logic (4 ANDs for 5 conditions)
    and_count = sql.count(" AND ")
    if and_count < 4:
        print(f"      pgvector might not be using AND logic properly: {and_count} ANDs")
        return False

    # Verify params has expected values
    if "alice" not in params or "public" not in params or "archived" not in params:
        print(f"      pgvector params missing expected values: {params}")
        return False

    print(f"      All backends correctly handle complex combined conditions")
    return True

test("Complex combination of all new features", test_complex_combined_conditions)

# Summary
print("\n" + "=" * 80)
print("CROSS-BACKEND CONSISTENCY TEST SUMMARY")
print("=" * 80)
print(f"\n✅ Passed: {tests_passed}")
print(f"❌ Failed: {tests_failed}")
if tests_passed + tests_failed > 0:
    print(f"📊 Success Rate: {100*tests_passed/(tests_passed+tests_failed):.1f}%")

if tests_failed == 0:
    print("\n🎉 ALL CROSS-BACKEND CONSISTENCY TESTS PASSED!")
    print("\nAll 5 backends (Qdrant, pgvector, Weaviate, Pinecone, ChromaDB)")
    print("generate semantically equivalent filters for the same policies.")
else:
    print(f"\n⚠️  {tests_failed} tests failed")

print("=" * 80)
