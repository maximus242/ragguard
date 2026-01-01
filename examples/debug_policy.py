#!/usr/bin/env python3
"""
Debugging a Policy
==================

This example shows how to debug when a policy isn't working as expected.
Useful when access is denied (or granted) unexpectedly.

Run with:
    python debug_policy.py
"""

from ragguard import Policy, PolicyEngine


def debug_policy():
    """Step-by-step policy debugging example."""

    # 1. Define a sample policy
    policy = Policy.from_dict({
        "version": "1",
        "rules": [
            {
                "name": "public-docs",
                "match": {"confidential": False},
                "allow": {"everyone": True}
            },
            {
                "name": "same-department",
                "allow": {"conditions": ["user.department == document.department"]}
            },
            {
                "name": "admin-access",
                "allow": {"roles": ["admin"]}
            }
        ],
        "default": "deny"
    })

    # 2. Define test cases
    test_user = {"id": "alice", "department": "engineering", "roles": ["developer"]}

    # Document that should be accessible (same department)
    doc_allowed = {"department": "engineering", "confidential": True}

    # Document that should be denied (different department, confidential)
    doc_denied = {"department": "finance", "confidential": True}

    # 3. Create engine and test
    engine = PolicyEngine(policy)

    print("=" * 70)
    print("RAGGuard Policy Debugger")
    print("=" * 70)
    print()

    print("USER CONTEXT:")
    print(f"  id: {test_user['id']}")
    print(f"  department: {test_user['department']}")
    print(f"  roles: {test_user['roles']}")
    print()

    # 4. Debug each test case
    for doc, expected in [(doc_allowed, True), (doc_denied, False)]:
        print("-" * 70)
        print(f"DOCUMENT: {doc}")
        print()

        # Evaluate
        result = engine.evaluate(test_user, doc)

        print("RULE-BY-RULE EVALUATION:")
        print()

        for i, rule in enumerate(policy.rules, 1):
            print(f"  Rule {i}: {rule.name}")

            # Check match conditions
            if rule.match:
                match_result = all(
                    doc.get(k) == v for k, v in rule.match.items()
                )
                print(f"    Match filter: {rule.match}")
                print(f"    Document values: {[f'{k}={doc.get(k)}' for k in rule.match]}")
                print(f"    Match result: {'PASS' if match_result else 'SKIP (no match)'}")
                if not match_result:
                    print()
                    continue

            # Check allow conditions
            if rule.allow.everyone:
                print(f"    Allow: everyone=True")
                print(f"    Result: ACCESS GRANTED")
                break

            if rule.allow.roles:
                user_roles = set(test_user.get("roles", []))
                required_roles = set(rule.allow.roles)
                has_role = bool(user_roles & required_roles)
                print(f"    Allow: roles={rule.allow.roles}")
                print(f"    User roles: {list(user_roles)}")
                print(f"    Result: {'ACCESS GRANTED' if has_role else 'No matching role'}")
                if has_role:
                    break

            if rule.allow.conditions:
                for cond in rule.allow.conditions:
                    print(f"    Condition: {cond}")

                    # Parse simple conditions for debugging
                    if "==" in cond:
                        left, right = cond.split("==")
                        left, right = left.strip(), right.strip()

                        # Get values
                        if left.startswith("user."):
                            left_val = test_user.get(left[5:])
                            left_name = f"user.{left[5:]} = {left_val!r}"
                        elif left.startswith("document."):
                            left_val = doc.get(left[9:])
                            left_name = f"document.{left[9:]} = {left_val!r}"
                        else:
                            left_val = left
                            left_name = left

                        if right.startswith("user."):
                            right_val = test_user.get(right[5:])
                            right_name = f"user.{right[5:]} = {right_val!r}"
                        elif right.startswith("document."):
                            right_val = doc.get(right[9:])
                            right_name = f"document.{right[9:]} = {right_val!r}"
                        else:
                            right_val = right
                            right_name = right

                        match = left_val == right_val
                        print(f"      {left_name}")
                        print(f"      {right_name}")
                        print(f"      Equal: {match}")
                        if match:
                            print(f"    Result: ACCESS GRANTED")
                            break
                else:
                    print(f"    Result: No conditions matched")

            print()

        print()
        print(f"FINAL RESULT: {'ACCESS GRANTED' if result else 'ACCESS DENIED'}")
        expected_text = "as expected" if result == expected else "UNEXPECTED!"
        print(f"Expected: {'GRANT' if expected else 'DENY'} ({expected_text})")
        print()

    # 5. Show generated filters
    print("=" * 70)
    print("GENERATED DATABASE FILTERS")
    print("=" * 70)
    print()
    print("These filters are applied during vector search:")
    print()

    for backend in ["chromadb", "qdrant", "pgvector"]:
        db_filter = engine.to_filter(test_user, backend=backend)
        print(f"{backend.upper()}:")
        if db_filter is None:
            print("  (no filter - access to all documents)")
        elif isinstance(db_filter, str):
            print(f"  {db_filter}")
        elif isinstance(db_filter, tuple):
            print(f"  SQL: {db_filter[0]}")
            print(f"  Params: {db_filter[1]}")
        else:
            import json
            print(f"  {json.dumps(db_filter, indent=4, default=str)}")
        print()


def main():
    print()
    debug_policy()
    print()
    print("TIP: Use 'ragguard filters policy.yaml --user user.json' for quick filter debugging")
    print()


if __name__ == "__main__":
    main()
