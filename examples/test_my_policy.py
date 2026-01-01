"""
Example: Testing RAGGuard Policies

This example shows how to test your access control policies using
the PolicyEngine.evaluate() method for basic testing.

For advanced testing features (PolicyTester, PolicyCoverageTester,
PolicySimulator), install ragguard-enterprise:
    pip install ragguard-enterprise
"""

from ragguard import Policy
from ragguard.policy import PolicyEngine
from ragguard.policy.explainer import QueryExplainer


def example_basic_testing():
    """Basic policy testing using PolicyEngine.evaluate()."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Policy Testing with PolicyEngine")
    print("=" * 60)

    # Load your policy
    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'admin_full_access',
                'allow': {'roles': ['admin']}
            },
            {
                'name': 'public_docs',
                'match': {'visibility': 'public'},
                'allow': {'everyone': True}
            },
            {
                'name': 'department_docs',
                'allow': {
                    'conditions': ['user.department == document.department']
                }
            }
        ],
        'default': 'deny'
    })

    # Create policy engine
    engine = PolicyEngine(policy)

    # Define test cases
    test_cases = [
        {
            "name": "admin_accesses_everything",
            "user": {"id": "alice", "roles": ["admin"]},
            "document": {"id": "doc1", "department": "engineering"},
            "expected": True,
            "description": "Admins should access all documents"
        },
        {
            "name": "everyone_accesses_public",
            "user": {"id": "bob", "roles": ["user"]},
            "document": {"id": "doc2", "visibility": "public"},
            "expected": True,
            "description": "Anyone should access public documents"
        },
        {
            "name": "user_denied_private",
            "user": {"id": "bob", "roles": ["user"]},
            "document": {"id": "doc3", "visibility": "private", "department": "sales"},
            "expected": False,
            "description": "Users should be denied access to private docs from other departments"
        },
        {
            "name": "department_match",
            "user": {"id": "charlie", "roles": ["engineer"], "department": "engineering"},
            "document": {"id": "doc4", "department": "engineering", "visibility": "internal"},
            "expected": True,
            "description": "Users should access docs from their own department"
        }
    ]

    # Run tests
    passed = 0
    failed = 0

    for test in test_cases:
        result = engine.evaluate(test["user"], test["document"])
        status = "PASS" if result == test["expected"] else "FAIL"

        if result == test["expected"]:
            passed += 1
        else:
            failed += 1

        print(f"\n{status}: {test['name']}")
        print(f"  Description: {test['description']}")
        print(f"  Expected: {'allow' if test['expected'] else 'deny'}, Got: {'allow' if result else 'deny'}")

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("All policy tests passed!")
    else:
        print(f"Some tests failed!")


def example_test_with_explain():
    """Test policy with explanations."""
    print("\n" + "=" * 60)
    print("Example 2: Policy Testing with Explanations")
    print("=" * 60)

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'role_based',
                'allow': {'roles': ['admin', 'manager']}
            }
        ],
        'default': 'deny'
    })

    engine = PolicyEngine(policy)
    explainer = QueryExplainer(policy)

    test_cases = [
        {"user": {"id": "alice", "roles": ["admin"]}, "document": {"id": "doc1"}, "expected": True},
        {"user": {"id": "bob", "roles": ["manager"]}, "document": {"id": "doc2"}, "expected": True},
        {"user": {"id": "charlie", "roles": ["user"]}, "document": {"id": "doc3"}, "expected": False}
    ]

    for test in test_cases:
        result = engine.evaluate(test["user"], test["document"])
        explanation = explainer.explain(test["user"], test["document"])

        # Find first passing rule
        matched_rule = "default"
        for rule_eval in explanation.rule_evaluations:
            if rule_eval.matched and rule_eval.passed:
                matched_rule = rule_eval.rule_name
                break

        status = "PASS" if result == test["expected"] else "FAIL"
        print(f"\n{status}: User {test['user']['id']} with roles {test['user']['roles']}")
        print(f"  Decision: {explanation.final_decision}")
        print(f"  Matched rule: {matched_rule}")
        print(f"  Reason: {explanation.reason}")


def example_filter_generation():
    """Test filter generation for different backends."""
    print("\n" + "=" * 60)
    print("Example 3: Filter Generation Testing")
    print("=" * 60)

    policy = Policy.from_dict({
        'version': '1',
        'rules': [
            {
                'name': 'department_access',
                'allow': {
                    'conditions': ['user.department == document.department']
                }
            }
        ],
        'default': 'deny'
    })

    engine = PolicyEngine(policy)

    users = [
        {"id": "alice", "department": "engineering"},
        {"id": "bob", "department": "finance"},
        {"id": "carol", "department": "hr"}
    ]

    backends = ["qdrant", "chromadb", "pgvector"]

    for user in users:
        print(f"\nUser: {user['id']} (department: {user['department']})")
        for backend in backends:
            try:
                filter_obj = engine.to_filter(user, backend)
                # Just check it doesn't throw
                print(f"  {backend}: Filter generated successfully")
            except Exception as e:
                print(f"  {backend}: Error - {e}")


def example_enterprise_features():
    """Show how to use enterprise testing features."""
    print("\n" + "=" * 60)
    print("Example 4: Enterprise Testing Features")
    print("=" * 60)

    try:
        from ragguard_enterprise.testing import PolicyTester, PolicyCoverageTester, PolicySimulator

        policy = Policy.from_dict({
            'version': '1',
            'rules': [
                {'name': 'admin', 'allow': {'roles': ['admin']}},
                {'name': 'public', 'match': {'visibility': 'public'}, 'allow': {'everyone': True}}
            ],
            'default': 'deny'
        })

        # Use PolicyTester for structured testing
        tester = PolicyTester(policy)
        tester.add_test("admin_access", {"id": "alice", "roles": ["admin"]}, {"id": "doc1"}, "allow")
        tester.add_test("public_access", {"id": "bob"}, {"visibility": "public"}, "allow")
        tester.add_test("denied", {"id": "charlie"}, {"visibility": "private"}, "deny")

        results = tester.run()
        PolicyTester.print_results(results, verbose=True)

        # Use PolicySimulator for comprehensive testing
        simulator = PolicySimulator(policy)
        report = simulator.simulate_access_matrix(
            users=[{"id": "alice", "roles": ["admin"]}, {"id": "bob", "roles": ["user"]}],
            documents=[{"id": "doc1", "visibility": "private"}, {"id": "doc2", "visibility": "public"}]
        )
        print("\nAccess Matrix:")
        for user_id, docs in report.items():
            print(f"  {user_id}: {docs}")

    except ImportError:
        print("Enterprise testing features require ragguard-enterprise:")
        print("  pip install ragguard-enterprise")
        print("\nEnterprise features include:")
        print("  - PolicyTester: Structured test case management")
        print("  - PolicyCoverageTester: Role and field coverage analysis")
        print("  - PolicySimulator: Access matrix simulation")
        print("  - Test suite creation from YAML/JSON")


if __name__ == "__main__":
    example_basic_testing()
    example_test_with_explain()
    example_filter_generation()
    example_enterprise_features()
